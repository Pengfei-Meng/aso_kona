# =============================================================================

# =============================================================================
import os, sys, time, copy, argparse, shutil, unittest, pdb

# =============================================================================
# External Python modules
# =============================================================================
import numpy, pickle
import numpy as np
# =============================================================================
# Extension modules
# =============================================================================
import warnings
from mpi4py import MPI

# Import PETSc so it is initialized on ALL Procs
import petsc4py 
petsc4py.init(args=sys.argv)

from baseclasses import *
from pyspline import *
from pygeo import *
from sumb import *
from pywarp import *
# from pyoptsparse import SNOPT, Optimization

# ================================================================
#                   INPUT INFORMATION  
grid_file = './INPUT/Euler/Euler_CRM_L2'     # './INPUT/L2' 
FFD_file =  './INPUT/FFD_small.fmt'    # '../INPUT/FFD_small.fmt'   

## ========== Negative Volume Error =========== 
# grid_file = './INPUT/mdo/mdo_tutorial_euler' 
# FFD_file =  './INPUT/mdo/CRM_ffd.fmt'

output_directory =  'singleRun'
cdcl_file = output_directory + '/cdcl'
problem_name     = 'l2'

Mach    = 0.65
Chord_ref =  1.0
Area_ref = 3.407014
Span_ref = 3.758150834 

CFL=1.5
CFLcoarse=1.25
MGcycle='3w'
MGstart=2 

aeroOptions = {
    # Common Parameters
    'gridFile':grid_file+'.cgns',
    'outputDirectory':output_directory,
        
    # Physics Parameters
    'equationType':'Euler',
    'resaveraging':'noresaveraging',
    'liftIndex':3,
    'loadImbalance':0.1,
    'loadbalanceiter':1,
    'isoSurface':{'shock':1.0, 'vx':-0.001},
    
    # Solver Parameters
    'nsubiter':5,
    'nsubiterTurb':5,
    'CFL':CFL,
    'CFLCoarse':CFLcoarse,
    'MGCycle':MGcycle,
    'MGStartLevel':MGstart,
    'nCyclesCoarse':500,
    'nCycles':1000,
    'minIterationNum':0,
    'useNKSolver':True,
    'nkswitchtol':1e-2,
    
    # Solution monitoring
    'monitorvariables':['cpu', 'resrho','resturb', 'cl','cdp','cdv', 'cmy', 'yplus'],
    'surfaceVariables':['cp','vx', 'vy','vz', 'yplus'],
    'numberSolutions':True,
    'writeVolumeSolution':False,
    'printIterations':False,
    'printTiming':True,
    
    # Convergence Parameters
    'L2Convergence':2e-6,
    'L2ConvergenceCoarse':1e-4,
    'adjointl2convergence':1e-8,
    'adpc':True,
    'adjointsubspacesize':200,
    'frozenturbulence':False,
    'approxpc':True,
    'viscpc':False,
    'asmoverlap':2,
    'outerpreconits':3,
    'setMonitor':False,
    
    'useMatrixFreedRdw':True,
    }


meshOptions = {'gridFile':grid_file+'.cgns',
               'warpType':'algebraic',
               'solidWarpType':'topo',
               'solidSolutionType':'linear',
               }

# ================================================================
#                    Set Global Communicator
# ================================================================
gcomm = MPI.COMM_WORLD

if gcomm.rank == 0:
    # remove output directory and contents if it exists
    print output_directory
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    # create fresh output directory
    os.makedirs(output_directory)

# ======================================================================
#         DVGeometry Setup
# ====================================================================== 

evalFuncs = ['cl','cd','cmy']
aeroProblem = AeroProblem(name=problem_name, mach=Mach, reynolds=5e6, 
                 areaRef=Area_ref, alpha=2.2033, chordRef=Chord_ref, reynoldsLength=Chord_ref,
                 xRef=1.2077, zRef=0.007669, T=298.15, evalFuncs=evalFuncs)
#aeroProblem.addDV('alpha', lower=0, upper=5.0, scale=0.1)

DVGeo = DVGeometry(FFD_file)
DVGeo.addGeoDVLocal('shape',lower=-.25, upper=.25, axis='z', scale=1.0)

CFDsolver = SUMB(comm=gcomm, options=aeroOptions)
CFDsolver.setDVGeo(DVGeo)

# CFDsolver.addLiftDistribution(50, 'y')
span = 3.758150834 
# pos = numpy.array([0.0235, 0.267, 0.557, 0.695, 0.828, 0.944])*span
# CFDsolver.addSlices('y', pos, sliceType='absolute')

mesh = MBMesh(gcomm, options=meshOptions)
CFDsolver.setMesh(mesh)


# pdb.set_trace()
# ======================================================================
#         DVConstraint Setup
# ====================================================================== 

DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)

DVCon.setSurface(CFDsolver.getTriangulatedMeshSurface())

# Le/Te constraints (only iHigh is constrained)
DVCon.addLeTeConstraints(0, 'iHigh')
lIndex = DVGeo.getLocalIndex(0)
DVCon.addLeTeConstraints(0, indSetA=[lIndex[0, 0, 0]], indSetB=[lIndex[0, 0, 1]])
                         
# Setup curves for ref_axis
LE_pt = numpy.array([.01, 0.0, 0.0])
break_pt = numpy.array([.8477, 1.11853, 0.0])
tip_pt = numpy.array([2.85680, 3.75816, 0.0])
root_chord = 1.689
break_chord = 1.03628
tip_chord = .3902497

x_le = numpy.array([[LE_pt[0] + 0.01*root_chord, LE_pt[1], LE_pt[2]],
                    [break_pt[0] + 0.01*break_chord, break_pt[1], break_pt[2]],
                    [tip_pt[0] + 0.01*tip_chord, tip_pt[1], tip_pt[2]]])

x_te = numpy.array([[LE_pt[0] + 0.99*root_chord, LE_pt[1], LE_pt[2]],
                    [break_pt[0] + 0.99*break_chord, break_pt[1], break_pt[2]],
                    [tip_pt[0] + 0.99*tip_chord, tip_pt[1], tip_pt[2]]])

DVCon.addVolumeConstraint(x_le, x_te, nSpan=25, nChord=30,
                          lower=1.0,upper=3, scaled=True)

# Add the same grid of thickness constraints with minimum bound of 0.25
DVCon.addThicknessConstraints2D(x_le, x_te, 25, 30, lower=0.25, scaled=True)


#================== Solving ===================
# ========= setAeroProblem should happen after DVCon defined ======= 

# Start the timing
gcomm.barrier()
start = time.clock()

# pdb.set_trace()

CFDsolver(aeroProblem)
# CFDsolver.setAeroProblem(aeroProblem)   

# on checking Thickness dCdX dimension
# geoSens = {}
# DVCon.evalFunctionsSens(geoSens, includeLinear=True)

# pdb.set_trace()


# if CFDsolver.comm.rank == 0:
#     sol = CFDsolver.getSolution()
#     cd = sol['cd']
#     cl = sol['cl']
#     cm = sol['cmy']

#     print 'Cd, Cl, Cm, ', cd, cl, cm

# CFDsolver._setupAdjoint()
# CFDsolver._setAeroDVs()
#================ Setting ones and zeros ================
RHS = CFDsolver.getStates()

ones1 = np.ones_like(RHS)
ones2 = np.ones_like(RHS)
ones3 = np.ones_like(RHS)

out1 = np.zeros(len(RHS))
out2 = np.zeros(len(RHS))
out3 = np.zeros(len(RHS))

# zeros1 = np.zeros_like(RHS)


#======== The influence of JvBwd, JvFwd to 
#======== objective Cd, dFdX  =====================
# out1 = CFDsolver.computeJacobianVectorProductBwd(resBar=zeros1.copy(), wDeriv=True) 


#==================== solveadjoint =====================

# CFDsolver._setupAdjoint()

# out2 = CFDsolver.sumb.computematrixfreeproductbwdfast(ones1.copy())
# pdb.set_trace()

# print 'norm(fast), ', np.linalg.norm(out2)

# # uncomment the following two lines, otherwise solveadjoint won't converge
# # objective = 'cd'
# # CFDsolver.computeJacobianVectorProductBwd(funcsBar={objective.lower():1.0}, wDeriv=True)
# # CFDsolver.computeJacobianVectorProductBwd(funcsBar={objective.lower():1.0}, xDvDeriv=True)
# out1 = CFDsolver.computeJacobianVectorProductBwd(resBar=zeros1.copy(), wDeriv=True) 
# print 'norm(bwd), ', np.linalg.norm(out1)




# CFDsolver.computeJacobianVectorProductBwd(resBar=RHS, xDvDeriv=True) 

## ----------------------------------------------------------------------------------------


# CFDsolver.computeJacobianVectorProductFwd(wDot=RHS, residualDeriv=True)

# CFDsolver.sumb.solveadjoint(ones1.copy(), out1, True)

# # dR1 = CFDsolver.computeJacobianVectorProductBwd(resBar=out1, wDeriv=True) 
# # print 'res1', np.linalg.norm(dR1-ones1)

# dR1 = CFDsolver.sumb.computematrixfreeproductbwdfast(out1.copy())
# print 'res1', np.linalg.norm(dR1-ones1)



# #================= solveadjointforrhs ==================
# uncomment the following two lines, otherwise solveadjointforrhs won't converge

# objective = 'cd'
# CFDsolver.computeJacobianVectorProductBwd(funcsBar={objective.lower():1.0}, wDeriv=True)
# CFDsolver.computeJacobianVectorProductBwd(resBar=RHS, wDeriv=True) 
# CFDsolver.computeJacobianVectorProductFwd(wDot=RHS, residualDeriv=True)

# out2 = CFDsolver.sumb.solveadjointforrhs(ones2, 1.e-6)
# out2 = CFDsolver.solveAdjointForRHS(ones2.copy(), relTol=1e-6)

# dR2 = CFDsolver.computeJacobianVectorProductBwd(resBar=out2, wDeriv=True)
# print 'res2', np.linalg.norm(dR2-ones2)/numpy.linalg.norm(ones2)


# # #================= solvedirectforrhs ==================
# # uncomment the following line, otherwise solvedirectforrhs won't converge

# dR1 = CFDsolver.computeJacobianVectorProductFwd(wDot=zeros1, residualDeriv=True)
# print 'norm(fwd), ', np.linalg.norm(dR1)

# out3 = CFDsolver.solveAdjointForRHS(ones3.copy(), relTol=1e-6)
# # # out3 = CFDsolver.sumb.solvedirectforrhs(ones3, 1.e-6)

# dR3 = CFDsolver.computeJacobianVectorProductFwd(wDot=out3, residualDeriv=True)
# print 'res3', np.linalg.norm(dR3-ones3)/numpy.linalg.norm(ones3)



# CFDsolver.sumb.solvedirect(ones1.copy(), out1, True)





# #=======================================================





# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# CFDsolver.solveAdjoint(aeroProblem, 'cd')
# # funcSens = {}
# # CFDsolver.evalFunctionsSens(aeroProblem, funcSens, ['cl', 'cd'])



# #== Now on why dCdX so accurate, does it involve lin_solve or adjoint_solve? ==
# #== No, it doesn't lin_solve or adjoint solve, discovered something else though ==== 
# #== Go Back to check lin_solve and adjoint_solve === 

# # =============== Linear Solve, Adjoint Solve Accuracy Check ============== # 
# # if CFDsolver.comm.rank == 0:
# #     print 'dir(CFDsovler.sumb) ', dir(CFDsolver.sumb)



# ##=============================================================## 


# #===== Now Check the accuracy of the solution by using computeMatrixFreeProductFwd
# #===== & computeMatrixFreeProductBwd =============================================
# # print 'solveDirectForRHS vs. solvedirect', np.linalg.norm(out1 - out4)
# # print 'solveAdjointForRHS vs. solveadjoint', np.linalg.norm(out2 - out3)

# # pdb.set_trace()

# # =============== Code is running, but accuracy suspicious ============== # 
# # # ========= The above is running great ==========
# # if CFDsolver.comm.rank == 0:
# #     print 'dRdXT -------'
# # rhs = CFDsolver.getStates()
# # rhs = np.random.rand(len(rhs))
# # dRdXT = CFDsolver.computeJacobianVectorProductBwd(resBar=rhs, xDvDeriv=True)

# if CFDsolver.comm.rank == 0:
#     print 'dRdX -------'
# shapeDat = 0.01*numpy.ones(72)   # ((72,))
# designDot = {'shape':shapeDat}     #, 'alpha':2.0
# # pdb.set_trace()
# dRdX = CFDsolver.computeJacobianVectorProductFwd(xDvDot=designDot, residualDeriv=True)

gcomm.barrier()
end = time.clock()
duration = end - start # in seconds
if CFDsolver.comm.rank == 0:
    print('\n Wall time: %4.4f\n'%duration)



# ##================= Stuff from __call__ ================##

# # CFDsolver.setAeroProblem(aeroProblem)

# # releaseAdjointMemory = kwargs.pop('relaseAdjointMemory', True)

# # if CFDsolver.curAP.sumbData.stateInfo is None:
# #     CFDsolver.resetFlow(aeroProblem, releaseAdjointMemory)


