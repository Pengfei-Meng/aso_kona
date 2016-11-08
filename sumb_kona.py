import numpy as np
import time
import shelve
from mpi4py import MPI
from sumb import SUMB
from kona.user import BaseVector, UserSolver
import math
import pdb, copy, pickle

class Error(Exception):
    """
    Format the error message in a box to make it clear this
    was a expliclty raised exception.
    """
    def __init__(self, message, comm=None):
        msg = '\n+'+'-'*78+'+'+'\n' + '| SUMB4KONA Error: '
        i = 15
        for word in message.split():
            if len(word) + i + 1 > 78: # Finish line and start new one
                msg += ' '*(78-i)+'|\n| ' + word + ' '
                i = 1 + len(word)+1
            else:
                msg += word + ' '
                i += len(word)+1
        msg += ' '*(78-i) + '|\n' + '+'+'-'*78+'+'+'\n'
        if comm == None:
            print(msg)
        else:
            if comm.rank == 0: print(msg)
        Exception.__init__(self)
        
def fixArrayShape(vector):
    vecShape = vector.shape
    if vecShape[0] == 1:
        vector = vector.reshape(vecShape[1])
    return vector
        
def dictionaryDot(xDict, yDict):
    # initialize the dot product as zero
    out = 0
    # loop over vector segments in the dictionary
    for key in xDict:
        # make sure that the key exists in both vector dictionaries
        if key not in yDict:
            raise Error('dictionaryDot(): \
                        Y vector is missing the \"%s\" key. \
                        Cannot perform dot product.'%key)
        # get the segment value out of the dictionary
        x = xDict[key]
        y = yDict[key]
        # if x and y are vectors, reshape them into row vectors
        if hasattr(x, "__len__") and hasattr(y, "__len__"):
            xShape = x.shape
            if xShape[0] == 1:
                x = x.reshape(xShape[1])
                y = y.reshape(xShape[1])
            else:
                x = x.reshape(xShape[0])
                y = y.reshape(xShape[0])
                    
        # add the dot product contribution from this vector segment
        out += np.dot(x, y)    
    # check if the result came out as an array
    if hasattr(out, "__len__"):
        # if so, make sure it's an array of one element
        if len(out) == 1:
            # output the element itself as the result
            return out[0]
        else:
            # otherwise, raise error because we got a non-scalar dot prod
            raise Error('dictionaryDot(): \
                        Inner product of X and Y returned an array! \
                        Check X and Y sizes. They should be 1-D vectors.')
    else:
        # if the result is already a scalar, just spit it out
        return out
        
def dictionaryNorm(X):
    return np.sqrt(dictionaryDot(X, X))

class SUMB4KONA(UserSolver):
    """
    design variables only shape parameters here
    """
        
    def __init__(self, objFun, DVCon, conVals, SUMB, outDir='', 
                 debugFlag=False, augFactor=0.0, solveTol=1e-6):
        # store pointers to the problem objects for future use        
        
        self.CFDsolver = SUMB
        self.MBMesh = self.CFDsolver.mesh
        self.aeroProblem = self.CFDsolver.curAP
        self.DVGeo = self.CFDsolver.DVGeo
        self.DVCon = DVCon
        # complain about objective function if it's not valid
        if objFun not in self.CFDsolver.sumbCostFunctions:
            raise Error('__init__(): \
                        Supplied objective function is not known to SUmb!')
        # store objective function strings
        self.objective = objFun
        # objective augmentation factor
        self.augFactor = augFactor
        # linear and adjoint RHS solve tolerance
        self.solveTol = solveTol
        # get geometric design variables
        self.geoDVs = self.DVGeo.getValues()
        # get aerodynamic design variables
        self.aeroDVs = {}
        for key in self.CFDsolver.aeroDVs:
            self.aeroDVs[key] = self.aeroProblem.DVs[key].value
        # merge design variables into single dictionary
        self.designVars = dict(self.geoDVs.items() + self.aeroDVs.items())
        # get dictionary of geometric constraints
        assert len(self.designVars) == 1, "Attention, code is for shape only design variables"
        geoCons = {}
        self.DVCon.evalFunctions(geoCons, includeLinear=True, config=None)
        # store the constraint offset values
        self.aeroConVals = {}
        self.geoConVals = {}
        num_eq = 0
        num_ineq = 0

        for key in conVals:
            if key in self.CFDsolver.sumbCostFunctions.keys():
                self.aeroConVals[key] = conVals[key]
                num_eq += 1

            elif key in geoCons:
                if hasattr(geoCons[key], '__len__'):
                    conShape = geoCons[key].shape
                    if conShape[0] == 1:
                        self.geoConVals[key] = conVals[key]*np.ones(conShape[1])
                        if key == 'lete': 
                            num_eq += conShape[1]
                        else:
                            num_ineq += conShape[1]
                    else:
                        self.geoConVals[key] = conVals[key]*np.ones(conShape[0])
                        if key == 'lete': 
                            num_eq += conShape[0]
                        else:
                            num_ineq += conShape[0]
                else:
                    self.geoConVals[key] = conVals[key]
                    if key == 'lete': 
                        num_eq += 1
                    else:
                        num_ineq += 1

            else:
                raise Error('__init__(): \
                            \'%s\' constraint is not valid. \
                            Check problem setup.')

        # merge aero and geo constraints into single constraint dictionary
        self.constraints = dict(self.geoConVals.items() + self.aeroConVals.items())
    

        # initialize optimization sizes
        self.num_state = self.CFDsolver.getStateSize()  
        # initialize the design norm check to prevent unnecessary mesh warping
        self.cur_design_norm = None

        num_pri = sum([len(v) for v in self.designVars.values()])

        num_sta = self.num_state

        self.local_state = self.CFDsolver.getStates()

        super(SUMB4KONA, self).__init__(num_design = num_pri, num_state = num_sta, 
            num_eq = num_eq, num_ineq = num_ineq)

        # set debug informationrm
        self.debug = debugFlag
        if self.get_rank() == 0:
            self.output = True
        else:
            self.output = False
        self.outDir = outDir
        # internal optimization bookkeeping
        self.iterations = 0
        self.totalTime = 0.
        self.startTime = time.clock()
        if self.output:
            file = open(self.outDir+'/kona_timings.dat', 'a')
            file.write('# SUMB4KONA iteration timing history\n')
            titles = '# {0:s}    {1:s}    {2:s}    {3:s}    {4:s}   {5:s}\n'.format(
                'Iter', 'Time (s)', 'Total Time (s)', 'Objective Val', 'Max Constraint', 'dual_norm')
            file.write(titles)
            file.close()
        
    def isNewDesign(self, design_vec): 
        design_dict = copy.deepcopy(self.designVars)
        # if design_vec.shape[0] == 1:
        #     design_vec = design_vec.reshape(design_vec.shape[1])
        design_dict['shape'] = design_vec
        if (dictionaryNorm(design_dict) != self.cur_design_norm):
            return True
        else:
            return False
    
    def updateDesign(self, design_vec):
        if self.output and self.debug: print('   |-- Updating design ...')

        # put design_vec back into design_dict 
        design_dict = copy.deepcopy(self.designVars)
        design_dict['shape'] = design_vec
        
        for DV in self.geoDVs:
            design_dict[DV] = fixArrayShape(design_dict[DV])
        # pass in GeoDV into the DVGeo object
        self.DVGeo.setDesignVars(design_dict)
        # pass in the AeroDV into aeroProblem object
        self.aeroProblem.setDesignVars(design_dict)
        # propagate changes through SUmb (this warps the mesh)
        self.CFDsolver.setAeroProblem(self.aeroProblem)
        # update the current design norm
        self.cur_design_norm = dictionaryNorm(design_dict)
        
    def fixGeoDVs(self, design_dict):
        for DV in self.geoDVs:
            if hasattr(design_dict[DV], '__len__'):
                design_dict[DV] = fixArrayShape(design_dict[DV])
        return design_dict

    def array2dict(self, in_vec):
        design_dict = copy.deepcopy(self.designVars)

        if in_vec.shape[0] == 1:
            in_vec = in_vec.reshape(in_vec.shape[1])

        design_dict['shape'] = in_vec
        return design_dict

################################################################################
#                          kona Toolbox Functions                            #
################################################################################
    

    def get_rank(self):
        return self.CFDsolver.comm.rank

    def allocate_state(self, num_vecs):
        return [BaseVector(len(self.local_state)) \
            for i in range(num_vecs)]        
                
    def eval_obj(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_obj() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design):
            self.updateDesign(at_design)

        self.CFDsolver.setStates(at_state.data)
        cost = 0
        # perform a residual evaluation to propagate the state change
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('eval_obj(): \
                        Residual norm is NaN. kona_state is wrong.')
        # calculate the cost functions and get solution dict
        sol = self.CFDsolver.getSolution()
        Obj = sol[self.objective.lower()]

        return (Obj, cost)
        
    def eval_residual(self, at_design, at_state, store_here):
        if self.output and self.debug: print('>> kona::eval_residual() <<')
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
 
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)

        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('eval_residual(): \
                        Residual norm is NaN. kona_state is wrong.')
        else:
            store_here.data = residual


    def eval_eq_cnstr(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_eq_cnstr() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem, 
                                              releaseAdjointMemory=False)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('eval_constraints(): \
                        Residual norm is NaN. kona_state is wrong.')
        # calculate the cost functions and get solution dict
        aeroFuncsDict = self.CFDsolver.getSolution()

        eq_cons = {}
        for key in self.aeroConVals:
            eq_cons[key] = aeroFuncsDict[key.lower()] - self.aeroConVals[key]
    
        # get and store the geometric constraints  ||  only lete
        geoConsDict = {}
        self.DVCon.evalFunctions(geoConsDict, includeLinear=True, config=None)
        eq_cons['lete'] = geoConsDict['lete'] - self.geoConVals['lete']

        return np.concatenate((np.array([eq_cons['cl']]), np.array([eq_cons['cmy']]), \
            eq_cons['lete'].flatten()))

    def eval_ineq_cnstr(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_ineq_cnstr() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)

        # get and store the geometric constraints
        geoConsDict = {}
        self.DVCon.evalFunctions(geoConsDict, includeLinear=True, config=None)

        ineq_cons = {}
        ineq_cons['thick'] = geoConsDict['thick'] - self.geoConVals['thick']
        ineq_cons['vol'] = geoConsDict['vol'] - self.geoConVals['vol']

        return np.concatenate((ineq_cons['thick'].flatten(), ineq_cons['vol'].flatten()))
                    
    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::multiply_dRdX() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)

        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('multiply_dRdX(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the complete multiplication via SUmb's matrix free routine
        designDot = self.array2dict(in_vec)

        out_vec.data = self.CFDsolver.computeJacobianVectorProductFwd(
            wDot=None, xDvDot=designDot, residualDeriv=True, funcDeriv=False)
        
    def multiply_dRdX_T(self, at_design, at_state, in_vec): 
        if self.output and self.debug: print('>> kona::multiply_dRdX_T() <<')
        
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)

        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)

        if math.isnan(resNorm):
            raise Error('multiply_dRdX_T(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the complete reverse product via SUmb's matrix free routine
        dRdxProd = self.CFDsolver.computeJacobianVectorProductBwd(
            resBar=in_vec.data, funcsBar=None, 
            wDeriv=False, xDvDeriv=True)
        # ensure proper vector shape for geoDVs
        dRdxProd = self.fixGeoDVs(dRdxProd)

        # only 'shape' design variables
        temp = np.array(dRdxProd.values(), dtype=float)
        return temp.flatten()
    
    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec): 
        if self.output and self.debug: print('>> kona::multiply_dRdU() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dRdU(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the multiplication via SUmb matrix-free routines
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=in_vec.data,
                                                   xDvDot=None,
                                                   residualDeriv=True,
                                                   funcDeriv=False)
        
    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec): 
        if self.output and self.debug: print('>> kona::multiply_dRdU_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dRdU_T(): \
                        Residual norm is NaN. kona_state is wrong.')
       
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductBwd(resBar=in_vec.data, wDeriv=True)


    def build_precond(self):
        pass
        
    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::apply_precond() <<')
        ########## THIS NEEDS TO BE FIXED LATER ##########
        out_vec.data = self.CFDsolver.globalNKPreCon(
                                  in_vec.data, 
                                  np.zeros(self.num_state))

        print 'apply_precond is being called! '

        return 1
        
    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::apply_precond_T() <<')
        out_vec.data = self.CFDsolver.globalAdjointPreCon(
                                  in_vec.data, 
                                  np.zeros(self.num_state))

        print 'apply_precond_T is being called! '

        return 1

    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCEQdX() <<')
        
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCEQdX(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the dC/dXdv multiplication for aerodynamic constraints
        designDot = self.array2dict(in_vec)

        aeroConProds = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=None,
                                                   xDvDot=designDot,
                                                   residualDeriv=False,
                                                   funcDeriv=True)
        # get the geometric constraint derivatives (these are cheap and small)
        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        # now loop over constraints and perform the multiplications
        # if this is an aero constraint, we already have the result
        eq_cons = {}

        for con in self.aeroConVals:
            eq_cons[con] = aeroConProds[con.lower()]

        # if this is a geometric constraint, we have work to do
        # 'lete' is the only equality constraints
        con = 'lete' 
        # start by zeroing out the result
        if hasattr(self.geoConVals[con], '__len__'):
            # if the constraint is a vector, set it to a zero vector
            conShape = self.geoConVals[con].shape
            if conShape[0] == 1:
                eq_cons[con] = np.zeros(conShape[1])
            else:
                eq_cons[con] = np.zeros(conShape[0])
        else:
            # otherwise set it to scalar zero
            eq_cons[con] = 0.0

        # loop over geometric design variables
        # only 'shape' design variable for now
        DV = 'shape'
        geoDV = in_vec
        dCdX = geoSens[con][DV]
        # check if the DV is a scalar or a vector
        if hasattr(geoDV, '__len__'):
            # get array sizes
            geoDVshape = geoDV.shape
            dCdXshape = dCdX.shape
            # reshape the design vector into a column vector
            if geoDVshape[0] == 1:
                geoDV = geoDV.reshape((geoDVshape[1], 1))
            else:
                geoDV = geoDV.reshape((geoDVshape[0], 1))
            # get the size again after the reshape
            geoDVshape = geoDV.shape
            # rehsape the jacobian to match the design vector
            if dCdXshape[1] != geoDVshape[0]:
                if dCdXshape[0] == geoDVshape[0]:
                    dCdX = np.transpose(dCdX)
                else:
                    raise Error('multiply_dCdX(): \
                                Constraint jacobian is not the \
                                right size! Check problem setup.')
        # perform the jacobian-vector multiplication
        eq_cons[con] = np.dot(dCdX, geoDV)
        # if the result is a vector, put it in row form
        return np.concatenate((np.array([eq_cons['cl']]), np.array([eq_cons['cmy']]), \
            eq_cons['lete'].flatten()))

        
    def multiply_dCEQdU(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCdU() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCdU(): \
                        Residual norm is NaN. kona_state is wrong.')
        # perform the {dC/dw} multiplication
        funcsDot = \
        self.CFDsolver.computeJacobianVectorProductFwd(wDot=in_vec.data,
                                                   xDvDot=None,
                                                   residualDeriv=False,
                                                   funcDeriv=True)
        # loop over constraints and perform the necessary tasks
        out_vec = {}
        # if this an aerodynamic constraint, get the result from SUmb
        for key in self.aeroConVals:
            out_vec[key] = funcsDot[key.lower()]
        # otherwise, if this is a geometric constraint, set it to zero
        # only 'lete' belongs to equaltiy constraints
        key = 'lete'
        if hasattr(self.geoConVals[key], '__len__'):
            out_vec[key] = \
                np.zeros(len(self.geoConVals[key]))
        else:
            out_vec[key] = 0.0

        return np.concatenate((np.array([out_vec['cl']]), np.array([out_vec['cmy']]), \
            out_vec['lete'].flatten()))
        
    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCdX_T() <<')

        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCdX_T(): \
                        Residual norm is NaN. kona_state is wrong.')
        # separate constraints into aero and geo
        aeroConDict = {}        # equality constraints 'cl' 'cmy' 'lete'
        geoConDict = {}

        aeroConDict['cl'] = in_vec[0]
        aeroConDict['cmy'] = in_vec[1]        

        key = 'lete'
        geoConDict[key] = in_vec[2:2+len(self.geoConVals[key])]
        # get the aerodynamic constriant contribution from SUmb        

        funcsDot = self.CFDsolver.computeJacobianVectorProductBwd(
                        resBar=None, funcsBar=aeroConDict,
                        wDeriv=False, xDvDeriv=True)

        # get the geometric constraint derivatives from DVCon on top
        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)
        
        # loop over the design variables and add the contributions
        DV = 'shape'
        out_vec = funcsDot[DV].flatten()

        con = 'lete'
        # get the necessary arraysn
        conVec = geoConDict[con]
        dCdX = geoSens[con][DV]
        # check if the constraint is a scalar or a vector
        if hasattr(conVec, '__len__'):
            # get array sizes
            conShape = conVec.shape
            dCdXshape = dCdX.shape
            # reshape the design vector into a column vector
            if conShape[0] == 1:
                conVec = conVec.reshape((conShape[1], 1))
            else:
                conVec = conVec.reshape((conShape[0], 1))
            # get the size again after the reshape
            conShape = conVec.shape
            # rehsape the jacobian to match the constraint vector
            # NOTE: mathematically, this will produce {dC/dXdv}^T
            if dCdXshape[1] != conShape[0]:
                if dCdXshape[0] == conShape[0]:  
                    dCdX = np.transpose(dCdX)
                else:
                    raise Error('multiply_dCdX_T(): \
                                Constraint jacobian is not the \
                                right size! Check problem setup.')
        out = np.dot(dCdX, conVec)
        # if the result is a vector, put it in row form
        if hasattr(out, '__len__'):
            outShape = out.shape
            outSize = outShape[0]*outShape[1]
            if outSize == 1:
                out = out[0, 0]
            else:
                if outShape[0] == 1:
                    out = out.reshape(outShape[1])
                elif outShape[1] == 1:
                    out = out.reshape(outShape[0])
                else:
                    raise Error('multiply_dCdX_T(): \
                                Result is not sized right!')
        out_vec += out
        return out_vec

    def multiply_dCEQdU_T(self, at_design, at_state, in_vec, out_vec):
        if self.output and self.debug: print('>> kona::multiply_dCdU_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update the state variables
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dCdU_T(): \
                        Residual norm is NaN. kona_state is wrong.')
        # separate constraints into aero and geo
        aeroConsDict = {}

        aeroConsDict['cl'] = in_vec[0]
        aeroConsDict['cmy'] = in_vec[1]   

        # perform the {dC/dw}^T multiplication
        out_vec.data = \
        self.CFDsolver.computeJacobianVectorProductBwd(resBar=None,
                                                   funcsBar=aeroConsDict,
                                                   wDeriv=True,
                                                   xDvDeriv=False)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdX() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
 
        # only geometric inequality constraints:  'thick'  'vol'  
        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        DV = 'shape'
        con1 = 'thick'
        con2 = 'vol'

        out_vec = {}
        dCdX1 = geoSens[con1][DV]        
        dCdX2 = geoSens[con2][DV]  

        # process the dimension for products
        out_vec[con1] = np.dot(dCdX1, in_vec)
        out_vec[con2] = np.dot(dCdX2, in_vec)        

        return np.concatenate((out_vec['thick'], out_vec['vol']))       


    def multiply_dCINdU(self, at_design, at_state, in_vec):

        return np.zeros(self.num_ineq)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        if self.output and self.debug: print('>> kona::multiply_dCINdX_T() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)    

        # only geometric inequality constraints:  'thick'  'vol'  
        geoSens = {}
        self.DVCon.evalFunctionsSens(geoSens, includeLinear=True)

        DV = 'shape'
        con1 = 'thick'
        con2 = 'vol'
        dCdX1 = geoSens[con1][DV]        
        dCdX2 = geoSens[con2][DV]  

        invec1 = in_vec[:len(self.geoConVals[con1])]
        invec2 = in_vec[len(self.geoConVals[con1]):]

        out_vec = np.dot(np.transpose(dCdX1), invec1) + np.dot(np.transpose(dCdX2), invec2)
        return out_vec


    def multiply_dCINdU_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.equals_value(0.0)

        
    def eval_dFdX(self, at_design, at_state):
        if self.output and self.debug: print('>> kona::eval_dFdX() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables   
        self.CFDsolver.setStates(at_state.data)
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('eval_dFdX(): '
                        'Residual norm is NaN. kona_state is wrong.')
        # get the design variable derivatives with the backward routine
        objFun = {}
        objFun[self.objective] = 1.0
        dJdx = self.CFDsolver.computeJacobianVectorProductBwd(
                        resBar=None, funcsBar=objFun,
                        wDeriv=False, xDvDeriv=True)
        # import pdb; pdb.set_trace()
        # ensure proper vector shape for geoDV
        dJdx = self.fixGeoDVs(dJdx)

        return np.array(dJdx.values(), dtype=float).flatten()
                        
    def eval_dFdU(self, at_design, at_state, store_here):
        if self.output and self.debug: print('>> kona::eval_dFdU() <<')
        # if the design has changed, update it
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # update state variables
        self.CFDsolver.setStates(at_state.data)
        # import pdb; pdb.set_trace()
        self.CFDsolver.sumb.computeresidualnk()
        residual = self.CFDsolver.getResidual(self.aeroProblem)
        resNorm = np.linalg.norm(residual)
        if math.isnan(resNorm):
            raise Error('multiply_dFdU(): '
                        'Residual norm is NaN. kona_state is wrong.')
        # get the state variable derivatives with the backward routine
        objFun = {}
        objFun[self.objective] = 1.0
        dJdw = self.CFDsolver.computeJacobianVectorProductBwd(
                                  resBar=None, funcsBar=objFun, 
                                  wDeriv=True, xDvDeriv=False)
        store_here.data = dJdw   # *10
        
    def init_design(self):
        # initial design is already set up outside of Kona
        # all we have to do is store the array into Kona memory
        if self.output and self.debug: print('>> kona::init_design() <<')
        init_design = self.designVars
        for key in self.geoDVs:
            if hasattr(self.geoDVs[key], '__len__'):
                size = self.geoDVs[key].shape
                if size[0] == 0:
                    init_design[key] = np.zeros(size[1])
                else:
                    init_design[key] = np.zeros(size[0])
            else:
                init_design[key] = 0.0
        for key in self.aeroDVs:
            if key is 'alpha':
                init_design[key] = 2.3153
            if key is 'mach':
                init_design[key] = 0.65

        designV = self.fixGeoDVs(init_design.copy())
        designV_out = np.array(designV.values(), dtype=float).flatten()

        self.updateDesign(designV_out)

        return designV_out

    def solve_nonlinear(self, at_design, store_here):
        if self.output and self.debug: print('>> kona::solve_nonlinear() <<')
        # update the design if it changed
        if self.isNewDesign(at_design): 
            self.updateDesign(at_design)
        # solve the problem for this design, then solve it
        if self.output and self.debug:
            print('   |-- Performing system solution ...')
        self.CFDsolver.resetFlow(self.aeroProblem)
        self.CFDsolver(self.aeroProblem, releaseAdjointMemory=False, writeSolution=False)  

        # store the state variables into the Kona memory array
        if self.output and self.debug:
            print('   |-- Saving states into Kona memory ...')
        store_here.data = self.CFDsolver.getStates()
        if self.output and self.debug:
            print('   |-- Assembling adjoint matrix and solver ...')
        
        # self.CFDsolver.releaseAdjointMemory()

        # check solution convergence and exit
        solveFailed = self.CFDsolver.comm.allreduce(
            self.CFDsolver.sumb.killsignals.fatalfail, op=MPI.LOR)
        if solveFailed:
            return -1
        else:
            return 1
        
    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, store_here):   
        if self.output and self.debug: 
            print('>> kona::solve_linear() <<')
            print('   |-- tol: %.4g'%rel_tol)
        
        if not self.CFDsolver.adjointSetup: 
            self.CFDsolver._setupAdjoint()

        globalInner = rhs_vec.inner(rhs_vec)
        globalNorm = np.sqrt(globalInner)

        if globalNorm == 0.0:
            store_here.data[:] = 0.
            return 1
        else:
            # solve {dR/dw} * kona_state[result] = kona_state[rhs]        
            # self.CFDsolver.sumb.solvedirect(rhs_vec.data, store_here.data, True)
            store_here.data = self.CFDsolver.solveDirectForRHS(rhs_vec.data, relTol=rel_tol)
            # check for convergence and return cost metric to Kona
            solveFailed = self.CFDsolver.comm.allreduce(
                self.CFDsolver.sumb.killsignals.adjointfailed, op=MPI.LOR)

            if solveFailed:
                return -1
            else:
                return 1
        
    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, store_here):
        if self.output and self.debug: 
            print('>> kona::solve_adjoint() <<')
            print('   |-- tol: %.4g'%rel_tol)
        # absTol = 1e-6

        if not self.CFDsolver.adjointSetup: 
            self.CFDsolver._setupAdjoint()

        # # calculate initial residual norm       
        globalInner = rhs_vec.inner(rhs_vec)
        globalNorm = np.sqrt(globalInner)

        if globalNorm == 0.0:
            store_here.data[:] = 0.
            return 1
        else:
            # calculate the relative tolerance we need to pass to 
            self.CFDsolver.sumb.solveadjoint(rhs_vec.data.copy(), store_here.data, True)

            # check for convergence and return cost metric to Kona
            solveFailed = self.CFDsolver.comm.allreduce(
                self.CFDsolver.sumb.killsignals.adjointfailed, op=MPI.LOR)

            if solveFailed:
                return -1
            else:
                return 1
    

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        if self.output and self.debug: print('>> kona::current_solution() <<')

        return " "     



