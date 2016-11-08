# =============================================================================
# Standard Python modules
# =============================================================================
import os, sys, time, copy, argparse, shutil, unittest, pdb, imp

# =============================================================================
# External Python modules
# =============================================================================
import numpy

# =============================================================================
# Extension modules
# =============================================================================
# import MPI
from mpi4py import MPI

# Import PETSc so it is initialized on ALL Procs
import petsc4py 
petsc4py.init(args=sys.argv)

# MDOLab Imports
from baseclasses import *
from pyspline import *
from pygeo import *
from sumb import *
from pywarp import *

# # Kona optimization library
# imp.load_source('kona', '/fasttmp/mengp2/HYDRA/mdolab/newKona/src/')

import kona

# UserFunction wrapper for SUMB solver
curDir = os.path.dirname(__file__)
moduleDir = os.path.join(curDir, '../')
sys.path.append(moduleDir)
from sumb_kona import *
# from sumb_vector import *

# ================================================================
#                    Set Global Communicator
# ================================================================
gcomm = MPI.COMM_WORLD

# ================================================================
#                   INPUT INFORMATION
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../OUT')
parser.add_argument("--mesh", type=str, default='L2', choices=['L0', 'L1', 'L2'])
parser.add_argument("--FFD", type=int, default=72, choices=[72, 192, 768])
parser.add_argument("--mach", type=float, default=0.65, choices=[0.65, 0.85])
parser.add_argument("--iter", type=int, default=3)
args = parser.parse_args()


# set mesh information
grid_file = '../../rsa4/INPUT/Euler/Euler_CRM_%s'%args.mesh
if args.mesh == 'L0':
    if args.mach == 0.65:
        alphaInit = 2.2033
    elif args.mach == 0.85:
        alphaInit = 0.9307
    else:
        raise Error('opt.py --> \
                    Wrong mach number. Must be 0.65 or 0.85!')
    CFL = 1.5
    CFLcoarse = 1.25
    MGcycle = '3w'
    MGstart = -1
elif args.mesh == 'L1' or args.mesh == 'L2':
    if args.mach == 0.65:
        alphaInit = 2.3153
        # alphaInit = 2.2
    elif args.mach == 0.85:
        alphaInit = 1.0244
    else:
        raise Error('opt.py --> \
                    Wrong mach number. Must be 0.65 or 0.85!')
    CFL = 1.5
    CFLcoarse = 1.25
    MGcycle = '3w'
    MGstart = 2
else:
    raise Error('opt.py --> \
                Wrong mesh type specified! Must be L0, L1 or L2.')
                
# set FFD information
if args.FFD == 72:
    FFD_file = '../../rsa4/INPUT/FFD_small.fmt'
elif args.FFD == 192:
    FFD_file = '../../rsa4/INPUT/FFD.fmt'
elif args.FFD == 768:
    FFD_file = '../../rsa4/INPUT/FFD_large.fmt'
else:
    raise Error('opt.py --> \
                Wrong FFD type specified! Must be 72, 192 or 768.')
                
# directory actions should be done in serial
if gcomm.rank == 0:
    # remove output directory and contents if it exists
    print args.output
    if not os.path.exists(args.output):
    #     shutil.rmtree(args.output)
    # create fresh output directory
        os.makedirs(args.output)

# MPI barrier to make sure nothing moves forward until the directories are done
gcomm.barrier()

# ================================================================
#               Set Options for each solver
# ================================================================

aeroOptions = {
    # Common Parameters
    'gridFile':grid_file+'.cgns',
    'outputDirectory':args.output,
        
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
    }

meshOptions = {'gridFile':grid_file+'.cgns',
               'warpType':'algebraic',
               'solidWarpType':'topo',
               'solidSolutionType':'linear',
               }

# Optimizer
optns = {
    'max_iter' : args.iter,
    'opt_tol' : 1.e-3,
    'feas_tol' : 1.e-3,        
    'info_file' : args.output+'/kona_info.dat',
    'hist_file' : args.output+'/kona_hist.dat',

    'homotopy' : {
        'lambda' : 0.0,
        'inner_tol' : 1e-2,
        'inner_maxiter' : 50,
        'nominal_dist' : 1.0,
        'nominal_angle' : 7.0*np.pi/180.,
    },

    'verify' : {
        'primal_vec'     : True,
        'state_vec'      : True,
        'dual_vec_eq'    : True,
        'dual_vec_in'    : True,
        'gradients'      : True,
        'pde_jac'        : True,
        'cnstr_jac_eq'   : True,
        'cnstr_jac_in'   : True,
        'red_grad'       : True,
        'lin_solve'      : True,
        'out_file'       : args.output+'/kona_verify.dat',
    },

}


# =====================================================
#        Setup Flow Solver and Mesh
# =====================================================
# initialize the flow solver.
CFDsolver = SUMB(comm=gcomm, options=aeroOptions)

# Define the aerodynamic problem
aeroProblem = AeroProblem(name='Euler_CRM',
                          # L0: 0.9307 @ M=0.85 | 2.2033 @ M = 0.65
                          # L1: 1.0244 @ M=0.85 | 2.3153 @ M = 0.65
                          alpha=alphaInit, 
                          beta=0.0, 
                          mach=args.mach,
                          P=30000.0,
                          T=255.5,
                          areaRef=3.407014,
                          chordRef=1.0,
                          spanRef=3.758150834,
                          xRef=1.20777,
                          yRef=0,
                          zRef=0.007669,
                          evalFuncs=['cd','cl','cmy'])
                          
# add the aerodynamic design variables
#aeroProblem.addDV('alpha', value=2.3153, name='alpha')
#aeroProblem.addDV('mach', value=0.65, name='mach')

# initialize the mesh
mesh = MBMesh(gcomm, options=meshOptions)
CFDsolver.setMesh(mesh)

# set up monitoring information for plots
span = 3.758150834 
# pos = numpy.array([0.0235, 0.267, 0.557, 0.695, 0.828, 0.944])*span
# CFDsolver.addSlices('y', pos, sliceType='absolute')
# CFDsolver.addLiftDistribution(50, 'y')

# =====================================================
#        Setup Design Variable Mapping Object
# =====================================================
# Create an instance of DVGeo
DVGeo = DVGeometry(FFD_file)

# set initial design variables for the geometry
nShape = DVGeo.addGeoDVLocal('shape', lower=-0.25, upper=0.25, axis='z')

# add the geometry object into the solver.
CFDsolver.setDVGeo(DVGeo)


# =====================================================
#        Setup Geometric Constraints
# =====================================================
# (Empty) DVConstraint Object
DVCon = DVConstraints()

# Set the geometry object for constraints
DVCon.setDVGeo(DVGeo)

# Set wing surface in constraints
wing = CFDsolver.getTriangulatedMeshSurface()
DVCon.setSurface(wing)

# Setup curves for ref_axis
LE_pt = numpy.array([0.0, 0.0001, 0.0])
break_pt = numpy.array([0.8477, 1.11853, 0.0])
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
                    
# # # Add thickness constraints
DVCon.addThicknessConstraints2D(x_le, x_te, 
   2, 3, lower=0.25, scaled=True, name='thick')     # 25 30             
   

# Add volume constraints
nConSpan = 2    
nConChord = 3   
DVCon.addVolumeConstraint(leList=x_le, teList=x_te, 
    nSpan=nConSpan, nChord=nConChord, lower=1.0, upper=3.0, 
    scaled=True, name='vol')

# Calculate the reference indexes for the leading and trailing edges
wing_vols = [0]
up_ind = []
low_ind = []
for ivol in wing_vols:
    sizes = DVCon.DVGeo.FFD.topo.lIndex[ivol].shape
    for j in xrange(sizes[1]): # Y is out the wing:
        # Only TE's are fixed
        up_ind.append(DVCon.DVGeo.FFD.topo.lIndex[ivol][-1,j,0])
        low_ind.append(DVCon.DVGeo.FFD.topo.lIndex[ivol][-1,j,-1])

    # Constraint just the leading edge at the root
    up_ind.append(DVCon.DVGeo.FFD.topo.lIndex[ivol][0, 0, 0])
    low_ind.append(DVCon.DVGeo.FFD.topo.lIndex[ivol][0, 0, -1])

# Add the leading and trailing edge constraints
DVCon.addLeTeConstraints(indSetA=up_ind, indSetB=low_ind, name='lete')

# pdb.set_trace()
CFDsolver.setAeroProblem(aeroProblem)

# =====================================================
#   Set-up Optimization Problem
# =====================================================

# Define equality constraint values
constraintValues = {'cl':0.5,
                    'cmy':-0.17,
                    'lete':0.0,
                    'vol':1.0,
                    'thick':0.25,                  
                    }
                    
# Set the regularization factor
augFactor = 0             # 0.0075
# Normalize the regularization factor by the design space sizes
#augFactor *= (72./float(args.FFD))

#======================== Opt ========================
# # # Initialize the Kona user function wrapper around SUMB

ASO_test = SUMB4KONA('cd', DVCon, constraintValues, CFDsolver, 
    debugFlag=False, outDir=args.output, augFactor=augFactor)


# # Start the timing
# # gcomm.barrier()
start = time.clock()

# Start the optimization
algorithm = kona.algorithms.PredictorCorrectorCnstrINEQ

optimizer = kona.Optimizer(ASO_test, algorithm, optns)
optimizer.solve()

# End the timing and write run duration it into file
gcomm.barrier()
end = time.clock()
duration = end - start  
if CFDsolver.comm.rank == 0:
    konaHist = open(args.output+'/kona_hist.dat', 'a')
    konaHist.write('\n Wall time: %4.4f\n'%duration)
    konaHist.close()

#======================== Verify =========================

# ASO_test = SUMB4KONA('cd', DVCon, constraintValues, CFDsolver, 
#     debugFlag=True, outDir=args.output, augFactor=augFactor)

# algorithm = kona.algorithms.Verifier
# optimizer = kona.Optimizer(ASO_test, kona.algorithms.Verifier, optns)
# optimizer.solve()
