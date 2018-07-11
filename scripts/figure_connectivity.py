import conpy
from mayavi import mlab

from config import fname

# Read the connectivity estimates
con = conpy.read_connectivity(fname.ga_con(condition='pruned'))
con_parc = conpy.read_connectivity(fname.ga_con(condition='parcelled'))

# Plot the degree map
stc = con.make_stc(summary='degree', weight_by_degree=False)
fig = mlab.figure(size=(300, 300))
brain = stc.plot(
    subject='fsaverage',
    hemi='both',
    background='white',
    foreground='black',
    time_label='',
    initial_time=0,
    smoothing_steps=5,
    figure=fig,
)
brain.scale_data_colormap(0, 1, stc.data.max(), True)
brain.add_annotation('aparc', borders=2)

# Save some views
mlab.view(0, 90, 450, [0, 0, 0])
mlab.savefig('../paper/figures/degree_rh.png', magnification=4)
mlab.view(180, 90, 450, [0, 0, 0])
mlab.savefig('../paper/figures/degree_lh.png', magnification=4)
mlab.view(180, 0, 450, [0, 10, 0])
mlab.savefig('../paper/figures/degree_top.png', magnification=4)
mlab.view(180, 180, 480, [0, 10, 0])
mlab.savefig('../paper/figures/degree_bottom.png', magnification=4)

# Plot the connectivity diagram
fig, _ = con_parc.plot(title='Parcel-wise Connectivity', facecolor='white',
                       textcolor='black', node_edgecolor='white',
                       colormap='plasma_r', vmin=0, show=False)
fig.savefig('../paper/figures/squircle.pdf', bbox_inches='tight')
