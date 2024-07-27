import numpy as np
import copy
import networkx as nx
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

import numpy as np
from tools.datautils import DataUtils
from tools.debug import Debug
from os.path import join, split 
import os , math
import matplotlib.pyplot as plt
from tools.filetools import FileTools
from graphplot.simmatrix import SimMatrixPlot
from connectomics.nettools import NetTools
import copy
from connectomics.network import NetBasedAnalysis
import numpy as np
import copy
import itertools
import networkx as nx
from scipy.linalg import eigh


NITERATIONS = 1000


dutils = DataUtils()
resultdir = join(dutils.ANARESULTSPATH,"connectomes_M_vs_S")
GROUP    = "Mindfulness-Project"
simplt   = SimMatrixPlot()
ftools   = FileTools(GROUP)
debug    = Debug()
nettools = NetTools()
nba      = NetBasedAnalysis()



data = np.load(join(resultdir,"simM_metab_struct.npz"))
struct_con_arr  = data["struct_con_arr"]
metab_con_arr   = data["metab_con_arr"]
metab_pvalues   = data["metab_pvalues"]
subject_id_arr  = data["subject_id_arr"]
session_arr     = data["session_arr"]
print(subject_id_arr)
print(session_arr)
sys.exit()
metab_con_arr[metab_pvalues>0.001] = 0


def compute_laplacian_spectrum_tf(matrix):
    num_nodes = matrix.shape[0]
    degree_matrix = tf.linalg.diag(tf.reduce_sum(matrix, axis=1))
    laplacian_matrix = degree_matrix - matrix
    eigenvalues = tf.linalg.eigvalsh(laplacian_matrix)
    return eigenvalues

def binarize_matrix_tf(matrix, threshold):
    abs_matrix = tf.abs(matrix)
    binarized_matrix = tf.where(abs_matrix < threshold, 0.0, tf.sign(matrix))
    return binarized_matrix

class ThresholdOptimizer(Model):
    def __init__(self, num_matrices, initial_thresholds):
        super(ThresholdOptimizer, self).__init__()
        self.thresholds = tf.Variable(initial_thresholds, dtype=tf.float32)

    def call(self, sim_matrices):
        spectra = []
        for i in range(len(sim_matrices)):
            threshold = self.thresholds[i]
            sim_matrix = sim_matrices[i]
            binarized_matrix = binarize_matrix_tf(sim_matrix, threshold)
            spectrum = compute_laplacian_spectrum_tf(binarized_matrix)
            spectra.append(spectrum)
        
        combined_spectra = tf.stack(spectra)
        std_deviation = tf.math.reduce_std(combined_spectra, axis=0)
        mean_std_deviation = tf.reduce_mean(std_deviation)
        return mean_std_deviation

def train_thresholds(sim_matrices, initial_thresholds, epochs=1000, lr=0.01):
    model = ThresholdOptimizer(len(sim_matrices), initial_thresholds)
    optimizer = Adam(learning_rate=lr)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            loss = model(sim_matrices)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(epochs):
        loss = train_step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.numpy()}")

    best_thresholds = model.thresholds.numpy()
    return best_thresholds, loss.numpy()

# Example usage
if __name__ == "__main__":
    # Assuming sim_matrices is a list of N 2D numpy arrays representing the similarity matrices
    sim_matrices = metab_con_arr

    # sim_matrices = [np.random.rand(5, 5) for _ in range(10)]  # Example similarity matrices
    initial_thresholds = np.linspace(0.65, 0.85, len(sim_matrices)).astype(np.float32)

# Convert sim_matrices to tensors for TensorFlow
    sim_matrices = [tf.convert_to_tensor(matrix, dtype=tf.float32) for matrix in sim_matrices]

    best_thresholds, min_std = train_thresholds(sim_matrices, initial_thresholds)
    print("Best Thresholds: ", best_thresholds)
    print("Minimum Standard Deviation: ", min_std)



th = [0.748989898989899, 0.759090909090909, 0.7691919191919192, 0.7772727272727272, 
0.7085858585858587, 0.8378787878787879, 0.7611111111111111, 0.7186868686868687, 0.7045454545454546, 
0.8075757575757576, 0.6601010101010101, 0.6702020202020202, 0.8156565656565656, 0.6560606060606061, 
0.7328282828282828, 0.8156565656565656, 0.6560606060606061, 0.6883838383838384, 0.6661616161616162, 
0.7287878787878788, 0.8136363636363636, 0.7287878787878788, 0.7873737373737374, 0.7308080808080808, 
0.7429292929292929, 0.7671717171717172, 0.6641414141414141, 0.8398989898989899, 0.6601010101010101, 
0.7914141414141413, 0.7792929292929293, 0.6924242424242424, 0.8277777777777777, 0.7994949494949495, 
0.7873737373737374, 0.7893939393939394, 0.8156565656565656, 0.8398989898989899, 0.7792929292929293, 
0.6782828282828283, 0.7873737373737374, 0.8358585858585859, 0.8297979797979798, 0.7792929292929293, 
0.8358585858585859, 0.7287878787878788, 0.8035353535353535, 0.759090909090909, 0.6661616161616162, 
0.7368686868686869, 0.7106060606060606, 0.843939393939394, 0.7813131313131313, 0.8035353535353535, 
0.7611111111111111, 0.7651515151515151, 0.753030303030303, 0.8459595959595959, 0.7227272727272728, 
0.843939393939394, 0.7166666666666667, 0.7388888888888889, 0.7368686868686869, 0.7065656565656566, 
0.6681818181818182, 0.841919191919192, 0.7227272727272728, 0.6782828282828283, 0.8378787878787879, 
0.7106060606060606, 0.7328282828282828, 0.7368686868686869, 0.6742424242424243, 0.8358585858585859, 
0.841919191919192, 0.751010101010101, 0.7813131313131313, 0.740909090909091, 0.7287878787878788, 
0.8217171717171717, 0.7954545454545454, 0.6762626262626262, 0.8338383838383838, 0.7813131313131313, 
0.652020202020202, 0.7429292929292929, 0.7348484848484849, 0.8116161616161616, 0.8196969696969697, 
0.7449494949494949, 0.740909090909091, 0.8378787878787879, 0.748989898989899, 0.7550505050505051, 
0.8358585858585859]

subject_id = ['S019' ,'S006' ,'S058' ,'S045' ,'S045' ,'S016' ,'S008' ,'S008' ,'S004' ,'S004',
 'S068' ,'S007' ,'S018' ,'S060' ,'S060' ,'S028' ,'S028' ,'S002' ,'S002' ,'S002'
 'S066' ,'S066' ,'S053' ,'S053' ,'S032' ,'S032' ,'S067' ,'S067' ,'S050' ,'S042',
 'S042' ,'S015' ,'S015' ,'S041' ,'S064' ,'S064' ,'S054' ,'S059' ,'S059' ,'S063',
 'S036' ,'S036' ,'S023' ,'S023' ,'S047' ,'S047' ,'S017' ,'S017' ,'S014' ,'S048',
 'S048' ,'S013' ,'S029' ,'S056' ,'S056' ,'S044' ,'S044' ,'S001' ,'S001' ,'S057',
 'S030' ,'S030' ,'S022' ,'S022' ,'S062' ,'S062' ,'S012' ,'S012' ,'S055' ,'S055',
 'S024' ,'S024' ,'S031' ,'S031' ,'S043' ,'S043' ,'S038' ,'S051' ,'S003' ,'S034',
 'S034' ,'S046' ,'S021' ,'S005' ,'S005' ,'S049' ,'S049' ,'S061' ,'S061' ,'S033',
 'S033' ,'S026' ,'S026' ,'S039' ,'S039']

session=['V1' ,'V2' ,'V2' ,'V2' ,'V1' ,'V2' ,'V2' ,'V3' ,'V2' ,'V1' ,'V1' ,'V2' ,'V2' ,'V2',
 'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V3' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1',
 'V2' ,'V2' ,'V1' ,'V2' ,'V1' ,'V1' ,'V2' ,'V1' ,'V1' ,'V2' ,'V1' ,'V1' ,'V2' ,'V1',
 'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V1' ,'V2' ,'V1' ,'V2' ,'V2' ,'V2' ,'V1' ,'V2',
 'V1' ,'V2' ,'V1' ,'V2' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1',
 'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V1' ,'V1' ,'V3' ,'V2' ,'V1' ,'V2' ,'V2' ,'V2',
 'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1' ,'V2' ,'V1']
subject_id = np.array(subject_id)
th = np.array(th)
session = np.array(session)

ids = np.argsort(subject_id)
subject_id = subject_id[ids]
session    = session[ids]
th         = th[ids]

data = list()
for i in range(len(th)):
    data.append([subject_id[i],session[i],th[i]])

data = np.array(data)