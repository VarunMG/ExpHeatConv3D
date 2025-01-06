import numpy as np
import dedalus.public as d3
import logging
import sys
logger = logging.getLogger(__name__)


def openFields_3D(file_name):
    with open(file_name, 'rb') as load_file:
        time = np.load(load_file)
        bFromFile = np.load(load_file)
        uFromFile = np.load(load_file)
        pFromFile = np.load(load_file)
    return time, bFromFile, uFromFile, pFromFile

def parse_input_file(contents):
    lines = contents.split('\n')
    params = []
    counter = 0
    for line in lines:
        if counter == 1: #R
            params.append(float(line))
        elif counter == 3: #Pr
            params.append(float(line))
        elif counter == 5: #alpha
            params.append(float(line))
        elif counter == 7: #yL
            params.append(float(line))
        elif counter == 9: #beta
            params.append(float(line))
        elif counter == 11: #ell
            params.append(float(line))
        elif counter == 13: #Nx
            params.append(int(line))
        elif counter == 15: #Ny
            params.append(int(line))
        elif counter == 17: #Nz
            params.append(int(line))
        elif counter == 19: #stop time
            params.append(float(line))
        elif counter == 21: #max timestep
            params.append(float(line))
        elif counter == 23: #initializing
            params.append(line)
        elif counter == 25 and line == 'True': #hex zeroing True
            params.append(True)
        elif counter == 25 and line == 'False': #hex zeroing False
            params.append(False)
        elif counter == 27 and line == 'True': #write True
            params.append(True)
        elif counter == 27 and line == 'False': #write False
            params.append(False)
        elif counter == 29:
            params.append(line)
        counter += 1
    #returns [Ra, Pr, alpha, beta, ell, Nx,Ny, Nz, stop time, max timestep, init (rand, load), hex zeroing (true or false), write (true or false), endString]
    return params

def isHex(arr):
    if np.all(arr[2::4,0::4,:] < 1e-16) and np.all(arr[2::4,1::4,:] < 1e-16) and np.all(arr[3::4,0::4,:] < 1e-16) and np.all(arr[3::4,1::4,:] < 1e-16) and np.all(arr[0::4,2::4,:] < 1e-16) and np.all(arr[0::4,3::4,:] < 1e-16) and np.all(arr[1::4,2::4,:] < 1e-16) and np.all(arr[1::4,3::4,:] < 1e-16):
        return True
    return False

def conductionState(alpha_coeff,beta,ell,z):
    return -1*alpha_coeff*ell**2*np.exp(-1*z/ell) + 0.5*beta*z**2 - alpha_coeff*ell*z+ alpha_coeff*ell**2*np.exp(-1/ell) + alpha_coeff*ell - 0.5*beta

def calcNu(b_var, alpha_coeff, beta, ell):
    bMeans = d3.Average(b_var,('x','y'))
    bottomAvg = bMeans(z=0)
    bottomAvg = bottomAvg.evaluate()
    bottomAvg = bottomAvg.allgather_data('g')
    bottomAvg = bottomAvg[0][0][0]
    return conductionState(alpha_coeff,beta,ell,0)/bottomAvg

#def calcNu(b_var, alpha_coeff, beta, ell):
    #bMeans = d3.Average(b_var,('x','y'))
    #bottomAvg = bMeans(z=0)
    #bottomAvg = bottomAvg.evaluate()['g']
    #bottomAvg = bottomAvg[0]
    #bottomAvg = bottomAvg[0]
    #return conductionState(alpha_coeff,beta,ell,0)/bottomAvg

def writeAuxData(fileName,tVals,AuxVals):
    with open(fileName,'wb') as AuxData:
        np.save(AuxData,tVals)
        np.save(AuxData,AuxVals)
    return 1

def getVerticalMeans(b_var):
    b_var.change_scales(1)
    temp_field = b_var.allgather_data('g')
    vert_means = np.mean(temp_field.T,axis=1)
    return vert_means

def writeVertMeans(fileName,time,b_var):
    vertMeans = getVerticalMeans(b_var)
    with open(fileName, 'wb') as vertMeanData:
        np.save(vertMeanData,time)
        np.save(vertMeanData,vertMeans)

def writeAllVertMeans(fileName,vertMeanData):
    with open(fileName, 'wb') as vertMeanFile:
        np.save(vertMeanFile,vertMeanData)
    return 1

def writeFields(fileName,time,b_var,u_var,p_var):
    b_var.change_scales(1)
    u_var.change_scales(1)
    p_var.change_scales(1)
    with open(fileName,'wb') as fluidData:
        np.save(fluidData,time)
        np.save(fluidData,b_var.allgather_data('g'))
        np.save(fluidData,u_var.allgather_data('g'))
        np.save(fluidData,p_var.allgather_data('g'))
    return 1

###################################
### Read params from input file ###
###################################

inputFile = open(sys.argv[1],"r")
inputs = inputFile.read()
inputFile.close()

params = parse_input_file(inputs)
R = params[0]
Pr = params[1]
alpha = params[2]
yL = params[3] #yL is redefined below
beta = params[4]
ell = params[5]
Nx = params[6] 
Ny = params[7] #Ny redefined below
Nz = params[8]
stop_sim_time = params[9]
max_timestep = params[10]
init = params[11]
hex_zeroing = params[12]
write = params[13]
endString = params[14]

######### Warning! overwriting input file!
yL = (2*np.pi)/(alpha*np.sqrt(3)) #hexagonal aspect ratio
Ny = Nx//2


logger.info('params are the following:')
logger.info('R = %f', R)
logger.info('Pr=%f', Pr)
logger.info('alpha=%0.16f', alpha)
logger.info('yL=%0.16f', yL)
logger.info('beta=%0.16f', beta)
logger.info('ell=%0.16f', ell)
logger.info('Nx=%i', Nx)
logger.info('Ny=%i', Ny)
logger.info('Nz=%i', Nz)
logger.info('end time=%f', stop_sim_time)
logger.info('max timestep=%f', max_timestep)
logger.info('init = ' + init)
logger.info('hex_zeroing = %i', hex_zeroing)
logger.info('write = %i', write)
logger.info('endstring = ' + endString)



##############



# Parameters
#alpha = 2.0
#Nx, Ny, Nz = 288, 32, 144
#R = 1e5
#Pr = 1
#stop_sim_time = 100
#max_timestep= 0.0005
#ell = 0.1
#beta = 1

alpha_coeff = 1/(ell*(1-np.exp(-1/ell))) 
dealias = 3/2
timestepper = d3.RK443
dtype = np.float64


# Bases
coords = d3.CartesianCoordinates('x','y','z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-np.pi/alpha, np.pi/alpha), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-1*yL/2, yL/2), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, 1), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=(xbasis,ybasis))
tau_b2 = dist.Field(name='tau_b2', bases=(xbasis,ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))

T_source = dist.Field(name='T_source', bases = (zbasis))

# Substitutions
#kappa = np.sqrt(16/(Ra*Pr))
#nu = np.sqrt((16*Pr)/Ra)
x, y, z = dist.local_grids(xbasis, ybasis,zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
dz = lambda A: d3.Differentiate(A, coords['z'])

T_source['g'] = alpha_coeff*np.exp(-z/ell) - beta


# Problem
problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(b) - div(grad_b) + lift(tau_b2) = - u@grad(b) + T_source")
problem.add_equation("dt(u) - Pr*div(grad_u) + grad(p) - Pr*R*b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("dz(b)(z=0) = 0")
problem.add_equation("u(z=0) = 0")
problem.add_equation("b(z=1) = 0")
problem.add_equation("u(z=1) = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if init == 'rand':
    b.fill_random('c', seed=42, distribution='normal', scale=1e-1) # Random noise
    b['g'] *= z*(1-z) #damp noise at walls
    #b['g'] += -z
    #b['g'] += 0.05*np.cos((1/2)*np.pi*(x-alpha))*np.sin(np.pi*z*alpha) #adding a perturbation
    b['g'] += conductionState(alpha_coeff,beta,ell,z) # Add conduction state background
elif init == 'load1':
    loadFile = '/grad/gudibanda/modelTesting/expHeating_3D/R1000000.0Pr1alpha2.0yL3.141592653589793ell0.1beta1Nx128Ny128Nz64_3D_T3.6621558_fields.npy'
    time, bArr, uArr, pArr = openFields_3D(loadFile)
    u.load_from_global_grid_data(uArr)
    b.load_from_global_grid_data(bArr)
    p.load_from_global_grid_data(pArr)
elif init == 'load2':
    loadFile = '/scratch/gudibanda/R100000.0Pr1.0alpha2.0yL3.141592653589793ell0.1beta1.0Nx128Ny128Nz64_3D_T150.0_runOutput/fluidData21.8069986.npy'
    time, bArr, uArr, pArr = openFields_3D(loadFile)
    u.load_from_global_grid_data(uArr)
    b.load_from_global_grid_data(bArr)
    p.load_from_global_grid_data(pArr)
elif init == 'load3':
    loadFile = '/scratch/gudibanda/R100000.0Pr1.0alpha1.1547005383792517yL3.141592653589793ell0.1beta1.0Nx128Ny128Nz64_3D_T150.0_zero_modes_all_from_prev_runOutput/fluidData3.7888674.npy'
    time, bArr, uArr, pArr = openFields_3D(loadFile)
    u.load_from_global_grid_data(uArr)
    b.load_from_global_grid_data(bArr)
    p.load_from_global_grid_data(pArr)
elif init == 'load4':
    loadFile = '/scratch/gudibanda/R100000.0Pr1.0alpha2.0yL1.8137993642342178ell0.1beta1.0Nx256Ny128Nz64_3D_T150.0_new_hexIC_test_runOutput/fluidData2.5081922.npy'
    time, bArr, uArr, pArr = openFields_3D(loadFile)
    u.load_from_global_grid_data(uArr)
    b.load_from_global_grid_data(bArr)
    p.load_from_global_grid_data(pArr)
elif init == 'zero_modes':
    loadFile = '/scratch/gudibanda/R100000.0Pr1.0alpha1.1547005383792517yL3.141592653589793ell0.1beta1.0Nx128Ny128Nz64_3D_T150.0_hexagon_IC_runOutput/fluidData4.838896.npy'
    time, bArr, uArr, pArr = openFields_3D(loadFile)
    u.load_from_global_grid_data(uArr)
    b.load_from_global_grid_data(bArr)
    p.load_from_global_grid_data(pArr)
elif init == 'hexIC_new_zero_modes':
    with open('hex_IC_cos_Nx128Ny128Nz64.npy','rb') as hexFile:
        hexArr = np.load(hexFile)
    b.load_from_global_grid_data(100*hexArr)
elif init == 'hexIC':
    with open('hexIC.npy','rb') as hexFile:
        hexArr = np.load(hexFile)
    b.load_from_global_grid_data(hexArr)
elif init == 'new_hexIC':
    k0 = alpha
    amp = 0.1
    Tcond0 = conductionState(alpha_coeff, beta, ell, 0)
    b['g'] = conductionState(alpha_coeff, beta, ell, z)+ Tcond0*(amp/3)*np.cos(np.pi*z/2)*(np.cos(k0*(x + y *np.sqrt(3))) + np.cos(k0*(x-y*np.sqrt(3))) + np.cos(2*k0*x))
    #pert = dist.Field(name='pert', bases=(ybasis))
    #pert['g'] = 0.1*np.sin(4*np.pi*y)
    #perturbed_state_op = b+pert
    #perturbed_state = perturbed_state_op.evaluate()
    #b['g'] = perturbed_state['g']
    #b['g'] += 0.1*np.sin(4*np.pi*y)


# Analysis
#snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
#snapshots.add_task(b, name='buoyancy')
#snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)


#solver.print_subproblem_ranks(dt=max_timestep)


##volume of box
# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
#flow.add_property(np.sqrt(d3.dot(u,u)), name='Re')
flow.add_property(b,name='TAvg')
#flow.add_property(b*u,name="Nu")
#flow_TAvg = flow.volume_integral('TAvg')
#flow.add_property(1 + (b*d3.dot(u,ez))/kappa,name="Nu")


# Main loop
startup_iter = 10
tVals = []
NuVals = []
TAvgVals = []
allVertMeans = []
box_volume = (2*np.pi/alpha)*yL*1


genFileName = 'R'+str(R)+'Pr'+str(Pr)+'alpha'+str(alpha)+'yL'+str(yL)+ 'ell'+str(ell)+'beta'+str(beta)+'Nx'+str(Nx)+'Ny'+str(Ny)+'Nz'+str(Nz)+'_3D_T' + str(stop_sim_time)+'_'+endString
auxDataFile = '/scratch/gudibanda/' + genFileName + '_auxData/'
NuFileName =  auxDataFile + genFileName + '_NuData.npy'
TAvgFileName = auxDataFile + genFileName + '_TAvgData.npy'
fluidDataFileName = '/scratch/gudibanda/'+genFileName + '_runOutput/fluidData'


try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        flow_TAvg = flow.volume_integral('TAvg')/box_volume
        flow_Nu = calcNu(b,alpha_coeff,beta,ell)
        tVals.append(solver.sim_time)
        NuVals.append(flow_Nu)
        TAvgVals.append(flow_TAvg)
        if hex_zeroing:
            b['c'][2::4,0::4,:] = 0
            b['c'][2::4,1::4,:] = 0
            b['c'][3::4,0::4,:] = 0
            b['c'][3::4,1::4,:] = 0
            b['c'][0::4,2::4,:] = 0
            b['c'][0::4,3::4,:] = 0
            b['c'][1::4,2::4,:] = 0
            b['c'][1::4,3::4,:] = 0

            p['c'][2::4,0::4,:] = 0
            p['c'][2::4,1::4,:] = 0
            p['c'][3::4,0::4,:] = 0
            p['c'][3::4,1::4,:] = 0
            p['c'][0::4,2::4,:] = 0
            p['c'][0::4,3::4,:] = 0
            p['c'][1::4,2::4,:] = 0
            p['c'][1::4,3::4,:] = 0

            u['c'][:,2::4,0::4,:] = 0
            u['c'][:,2::4,1::4,:] = 0
            u['c'][:,3::4,0::4,:] = 0
            u['c'][:,3::4,1::4,:] = 0
            u['c'][:,0::4,2::4,:] = 0
            u['c'][:,0::4,3::4,:] = 0
            u['c'][:,1::4,2::4,:] = 0
            u['c'][:,1::4,3::4,:] = 0

            #print(b['c'].shape)
            #coeff_arr = b.allgather_data('c')
            #logger.info(isHex(coeff_arr))

        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%0.16f, dt=%0.16f, Nu=%0.16f, <T>= %0.16f'  %(solver.iteration, solver.sim_time, timestep, flow_Nu, flow_TAvg))#, flow_Nu)) #, flow_Nu, flow_TAvg))
        if (solver.iteration-1) % 100 == 0 and write:
            writeAuxData(NuFileName,tVals,NuVals)
            writeAuxData(TAvgFileName,tVals,TAvgVals)
            fileName = fluidDataFileName + str(round(10000000*solver.sim_time)/10000000) + '.npy'
            write_output = writeFields(fileName,solver.sim_time,b,u,p)
            #if write == 0:
                #print('fields are not writing')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

fileName = fluidDataFileName + str(round(10000*solver.sim_time)/10000) + '.npy'
writeFields(fileName,solver.sim_time,b,u,v)
writeNu(NuFileName,tVals,NuVals)


#writeNu(NuFileName,tVals,NuVals)

