
qiskit_code = '''

import os
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import partial_trace
from qiskit.visualization import plot_bloch_multivector

import numpy as np
import plotly.graph_objects as go

# Create a directory to save the plots if it doesn't exist
save_dir = 'quantum_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create a Quantum Circuit with 2 qubits
qc = QuantumCircuit(2)

# Apply the X gate to qubit 0
qc.x(0)

# Apply the Hadamard gate to both qubits
qc.h(0)
qc.h(1)

# Apply a CNOT gate, with qubit 0 as control and qubit 1 as target
qc.cx(0, 1)
'''

viz_code = '''

# Simulate the circuit using the statevector simulator
simulator = Aer.get_backend('statevector_simulator')
state = simulator.run(transpile(qc, simulator)).result().get_statevector()

# Print the state vector
print('Statevector:', state)
# Save the Bloch sphere visualization (state of the qubits)
bloch_plot = plot_bloch_multivector(state)
bloch_plot.savefig(os.path.join(save_dir, 'bloch_sphere.png'))

# Function to compute the Bloch vector from a qubit's reduced density matrix
def bloch_vector(density_matrix):
    # Convert the DensityMatrix to a NumPy array for element access
    density_matrix_np = density_matrix.data
    
    bx = 2 * np.real(density_matrix_np[0, 1])
    by = 2 * np.imag(density_matrix_np[1, 0])
    bz = np.real(density_matrix_np[0, 0] - density_matrix_np[1, 1])
    return np.array([bx, by, bz])

# Function to add fixed lines to the Bloch sphere
def add_fixed_lines(fig):
    t = np.linspace(0, 2 * np.pi, 100)
    
    # Add lines for x-y plane
    fig.add_trace(go.Scatter3d(x=np.cos(t), y=np.sin(t), z=np.zeros_like(t),
                               mode='lines', line=dict(color='white', width=2)))
    
    # Add lines for y-z plane
    fig.add_trace(go.Scatter3d(x=np.zeros_like(t), y=np.cos(t), z=np.sin(t),
                               mode='lines', line=dict(color='white', width=2)))
    
    # Add lines for x-z plane
    fig.add_trace(go.Scatter3d(x=np.cos(t), y=np.zeros_like(t), z=np.sin(t),
                               mode='lines', line=dict(color='white', width=2)))
    
    # X-axis line (from -1 to 1 on the X axis, through the origin)
    fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0],
                               mode='lines', line=dict(color='white', width=4)))

    # Y-axis line (from -1 to 1 on the Y axis, through the origin)
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0],
                               mode='lines', line=dict(color='white', width=4)))

    # Z-axis line (from -1 to 1 on the Z axis, through the origin)
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1],
                               mode='lines', line=dict(color='white', width=4)))

# Function to create and save individual Bloch spheres for each qubit
def plot_bloch_sphere(qubit, bloch_vec, filename):
    # Set up a 1x1 subplot for one Bloch sphere
    fig = go.Figure()

    # Define the custom color gradient (pink, orange, white)
    custom_colorscale = [
        [0.0, '#D3288F'],  # Pink
        [0.5, '#FF9742'],  # Orange
        [1.0, '#F2F2F2']   # White
    ]

    # Add Bloch sphere for the qubit
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    
    fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale=custom_colorscale, showscale=False))
    
    # Add vector for the qubit
    fig.add_trace(go.Scatter3d(x=[0, bloch_vec[0]], y=[0, bloch_vec[1]], z=[0, bloch_vec[2]],
                               marker=dict(size=[0, 7], color='red'),
                               line=dict(width=30), name=f'Qubit {qubit}'))

    fig.add_trace(go.Scatter3d(x=[bloch_vec[0]], y=[bloch_vec[1]], z=[bloch_vec[2]],
                           mode='markers',
                           marker=dict(size=10, color='blue', symbol='circle'),
                           name=f'Qubit {qubit} end'))  
    # Add fixed lines to split the Bloch sphere into quadrants
    add_fixed_lines(fig)

    # Set up layout for the subplot with grid lines white but not persistent
    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            bgcolor='#14141A',  # Set the 3D scene background to black
            xaxis=dict(color='white', showgrid=False, zeroline=True, zerolinecolor='white', showspikes=False, backgroundcolor='#14141A'),
            yaxis=dict(color='white', showgrid=False, zeroline=True, zerolinecolor='white', showspikes=False, backgroundcolor='#14141A'),
            zaxis=dict(color='white', showgrid=False, zeroline=True, zerolinecolor='white', showspikes=False, backgroundcolor='#14141A')
        ),
        title=dict(
            text=f'Interactive 3D Bloch Sphere for Qubit {qubit}',
            x=0.5,  # Center the title
            xanchor='center',  # Ensures it is centered properly
            yanchor='top'  # Aligns the title at the top
        ),
        paper_bgcolor='#14141A',  # Set the overall paper background to black
        font_color='white',  # Set font color to white
        showlegend=False
    )

    # Save plot to file if filename is provided
    if filename:
        fig.write_html(filename)

    # Show the interactive plot
    fig.show()

# Compute the reduced density matrices for each qubit
density_matrix = np.outer(state.data, np.conj(state.data))
reduced_rho_0 = partial_trace(density_matrix, [1])  # Trace out qubit 1 to get qubit 0's density matrix
reduced_rho_1 = partial_trace(density_matrix, [0])  # Trace out qubit 0 to get qubit 1's density matrix

# Compute the Bloch vectors for each qubit
bloch_vec_0 = bloch_vector(reduced_rho_0)
bloch_vec_1 = bloch_vector(reduced_rho_1)

# Plot and save each qubit's Bloch sphere in separate HTML files
plot_bloch_sphere(0, bloch_vec_0, os.path.join(save_dir, 'qubit_0_bloch_sphere.html'))
plot_bloch_sphere(1, bloch_vec_1, os.path.join(save_dir, 'qubit_1_bloch_sphere.html'))

# Add measurement operations and simulate using qasm simulator
qc.measure_all()

# Simulate with the QASM simulator
simulator = Aer.get_backend('qasm_simulator')
tqc = transpile(qc, simulator)
result = simulator.run(tqc, shots=1000).result()

# Get the result counts
counts = result.get_counts()

# Print the measurement counts
print('Result counts:', counts)

'''