from vedo import *
from vedo.pyplot import plot, histogram, show
import numpy as np
import trimesh
from ScalarTVDecomp import *
################################################################
# GLOBAL CONTEXT
plt = Plotter()
to_eigen = 0
from_eigen = 0

################################################################

################################################################
# MESH PROCESSING FUNCTIONS

def dig_neighbours(M, curr_v, depth, max_depth):
    '''
    Recursively finds the vertices in the neighbourhood of degree depth

    :param M: mesh
    :param curr_v: current vertex we are digging into
    :param depth: current depth of the neighbourhood search
    :param max_depth: the max depth of the neighbourhood search
    :return: the neighbours of this vertex
    '''
    v_neighbours = []
    v_neighbours.append(curr_v)
    if depth == max_depth:
        return v_neighbours
    else:
        # get neighbours
        neighbours = M.vertex_neighbors[curr_v]
        for n in neighbours:
            nn = dig_neighbours(M, n, depth + 1, max_depth)
            v_neighbours.extend(nn)
        return v_neighbours


def blotching(M, VERTS, blotch_num=3, min_degrees=1, max_degrees=5):
    '''
    Given a mesh M, creates a colour matrix of blotches of colour onto
    a given set of vertices then return both the |V| x (R, G, B) colour
    matrix and the indicator function of where the colours are in the mesh

    :param VERTS: The vertex list of the mesh
    :param blotch_num: The
    :param min_degrees:
    :param max_degrees:
    :return:
    '''
    # Create a colour scheme that adds blotches of colour onto a mesh
    dim_v = VERTS.shape[0]
    vertices = np.random.randint(0, dim_v, size=(blotch_num, 1))
    colour_place_holder = np.ones_like(VERTS) * 255
    indicator_signal = np.zeros((dim_v, 1))
    for v in vertices:
        v_targets = []
        v_targets.append(v)
        # generate a random colour
        rnd_colour = np.random.randint(0, 255, size=(1, 3))
        # pick a degree of neighbours
        degree_neighbourhood = np.random.randint(min_degrees, max_degrees)
        v_targets.extend(dig_neighbours(M, int(v), 1, degree_neighbourhood))
        # rnd_colour = np.matlib.repmat(rnd_colour, len(v_targets), 1)
        for target in v_targets:
            colour_place_holder[target, :] = rnd_colour
            indicator_signal[target] = 1
    return colour_place_holder, indicator_signal


def get_colours_v2(v, scale=1):
    '''
    Computes the colour mask based on the eigen signal

    :param v: summed eigen signal for colours
    :param scale: bumps up visibility of colours by scaling the eigen signal
    :return: A |V| x (R, G, B) colour matrix
    '''
    dim_v = v.shape[0]
    v = np.reshape(v, (dim_v, 1))
    lower_bound = np.ones_like(v) * 255
    colours = np.ones((dim_v, 3)) * 255
    colours = np.minimum(lower_bound, colours * v * scale)
    colours = np.floor(colours)
    return colours.astype(np.uint8)

def posneg_vec_normalize(v):
    min_val = np.min(v)
    new_v = (v + np.abs(min_val)) / np.linalg.norm(v)
    return new_v

#################################################################

'''
def onLeftClick(evt):
    if not evt.actor: return
    cpt = vector(evt.picked3d) + [0,0,1]
    printc("Added point:", precision(cpt[:2],4), c='g')
    cpoints.append(cpt)
    update()

def onRightClick(evt):
    if not evt.actor or len(cpoints)==0: return
    p = cpoints.pop() # pop removes from the list the last obj
    plt.actors.pop()
    printc("Deleted point:", precision(p[:2], 4), c="r")
    update()

def update():
    global spline, points
    plt.remove([spline, points])
    points = Points(cpoints, r=8).c('violet').alpha(0.8)
    spline = None
    if len(cpoints)>2:
        spline = Spline(cpoints, closed=True).c('yellow').alpha(0.8)
        # spline.ForceOpaqueOn()  # VTK9 has problems with opacity
        # points.ForceOpaqueOn()
    plt.add([points, spline])
'''

def keyfunc(evt):
    #global spline, points, cpoints
    if evt.keyPressed == 'c':
        plt.resetCamera()
        printc("==== pressed c | Reset Canera ====", c="r")
    elif evt.keyPressed == 'y':
        # If the use pressed y then we will recolour the mesh to represent
        # the eigen values selected.
        if(from_eigen <= to_eigen):
            print("from eigen: ", from_eigen, "to eigen: ", to_eigen)
            recon = np.sum(phi[:, from_eigen:to_eigen], 1)
            recon_normalized = posneg_vec_normalize(recon) * 10
            #recon_colours = get_colours_v2(recon_normalized, 10)
            mesh.cmap('rainbow', recon_normalized)
            mesh_90.cmap('rainbow', recon_normalized)
            mesh_180.cmap('rainbow', recon_normalized)
            mesh_270.cmap('rainbow', recon_normalized)
            plt.show(mesh, mesh_90, mesh_180, mesh_270, points)
            printc("==== pressed y | Filtered ====", c="r")
        else:
            printc("==== pressed y | ERROR make sure green point is before red point ====", c="r")
    elif evt.keyPressed == 'r':
        # reset the colour of the mess back to the original
        mesh.cmap('rainbow', colour_vec)
        mesh_90.cmap('rainbow', colour_vec)
        mesh_180.cmap('rainbow', colour_vec)
        mesh_270.cmap('rainbow', colour_vec)
        plt.show(mesh, mesh_90, mesh_180, mesh_270, points)
        printc("==== pressed r | reseted the colour map to original ====", c="r")
    else:
        printc('key press:', evt.keyPressed)



############################################################

#t = """Click to add a point
#Right-click to remove
#Press c to clear points
#Press s to save to file"""
#instrucs = Text2D(t, pos='bottom-left', c='white', bg='green', font='Quikhand', s=0.9)

#plt.show(pic, instrucs, axes=True, bg='blackboard').close()

#######################################################################################
# First load the mesh
M = trimesh.load("./example_meshes/bob.off")
#M = trimesh.load("./example_meshes/decompose/armadillo.obj")
V = np.array(M.vertices)*15 # scale the mesh so it looks larger in hte visualization
F = np.array(M.faces)
mesh = Mesh([V, F])
mesh_90 = mesh.clone()
mesh_180 = mesh.clone()
mesh_270 = mesh.clone()
mesh_90.rotateY(90)
mesh_180.rotateY(180)
mesh_270.rotateY(270)
mesh_90.pos(30, 0, 0)
mesh_180.pos(60, 0, 0)
mesh_270.pos(90, 0, 0)



# Second we need to add colour to the mesh
[colour_mat, indicator_func] = blotching(M, V, 6, 5, 10)
colour_vec = np.sum(colour_mat, 1)
mesh.cmap("rainbow", colour_vec)
mesh_90.cmap("rainbow", colour_vec)
mesh_180.cmap("rainbow", colour_vec)
mesh_270.cmap("rainbow", colour_vec)

############################################################
# Perform the TV spectral decomposition of the scalar indicator
# function

# Set the hyper parameters
alpha = 1
alpha_scale = 0.01
n_comp = 40

# init decomposer
mySpecTVDecomposer = ScalarTVDecomp(M, indicator_func, n_comp, alpha, alpha_scale)

# decompose and get eigen signals
phi = mySpecTVDecomposer.decompose_scalar()
plotting_signal = mySpecTVDecomposer.GetTVSignal(phi)
#plotting_signal = np.arange(40)

############################################################
# Setting up the points and sliders for visualizing the eigen
# signals
z_offest = -10
x_offset = 3
y_offset = 3
y_scaler = 50
p1 = Point([3, 3, z_offest], c='red')
p2 = Point([3, 3, z_offest], c='green')
points = [p1, p2]

def slider_1(widget, event):
    value = widget.GetRepresentation().GetValue()
    y_val = plotting_signal[int(np.floor(value))] * y_scaler
    # print("slider 1 y: ", y_val)
    x_val = int(np.floor(value))
    global to_eigen
    to_eigen = x_val
    points[0].y(y_val + y_offset)  # set y coordinate position
    points[0].x(x_val + x_offset)

def slider_2(widget, event):
    value = widget.GetRepresentation().GetValue()
    y_val = plotting_signal[int(np.floor(value))] * y_scaler
    # print("slider 2 y: ", y_val)
    x_val = int(np.floor(value))
    global from_eigen
    from_eigen = x_val
    points[1].y(y_val + y_offset)  # set y coordinate position
    points[1].x(x_val + x_offset)

######################################################
# Defining the plot and adding interaction events

plt.addCallback('KeyPress', keyfunc)

plt.addSlider3D(
    slider_1,
    pos1=[-5, 0, z_offest],
    pos2=[-5, 40,z_offest],
    xmin=0,
    xmax=39,
    value=0,
    s=0.01,
    t=2,
    c="r",
    rotation=45,
    title="To eigen component",
)

plt.addSlider3D(
    slider_2,
    pos1=[-10, 0, z_offest],
    pos2=[-10, 40, z_offest],
    xmin=0,
    xmax=39,
    value=0,
    s=0.01,
    t=2,
    c="r",
    rotation=45,
    title="From eigen component",
)


x = np.arange(40)
y = plotting_signal

p = plot(x, y,
         ma=0.2,               # const. marker alpha
         lw=0,                 # no line width
         aspect=1,           # plot aspect ratio
        )

p.pos(4, 4, z_offest)


plt.show(mesh, mesh_90, mesh_180, mesh_270, points, p, "Scalar Spectral TV Decomp ", axes=False)

