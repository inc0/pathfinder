# %%
import glob
import pygamer
from tqdm import tqdm
from p_tqdm import p_map



all_proteins = glob.glob("/datasets/bigbind/BigBindV1/PPARG_HUMAN_229_505_0/*pocket.pdb")

def process_protein(p):
    mesh = pygamer.readPDB_molsurf(p)
    components, orientable, manifold = mesh.compute_orientation()
    mesh.correctNormals()
    return mesh, components, orientable, manifold

res = process_protein(all_proteins[0])

# %%
