from mayavi import mlab
import mne
from mne.bem import _fit_sphere
import conpy

from config import fname, subjects

mne.set_log_level('ERROR')

fwd = mne.read_forward_solution(fname.fwd(subject=subjects[0]))
fwd_r = mne.read_forward_solution(fname.fwd_r(subject=subjects[0]))
trans = fname.trans(subject=subjects[0])

fig1 = mne.viz.plot_alignment(fwd['info'], trans=trans, src=fwd_r['src'], meg='sensors', surfaces='white')
fig1.scene.background = (1, 1, 1)  # white
fig1.children[-1].children[0].children[0].glyph.glyph.scale_factor = 0.008
mlab.view(135, 120, 0.3, [0.01, 0.015, 0.058])
mlab.savefig('../paper/figures/forward1.png', magnification=4, figure=fig1)

radius, center = _fit_sphere(fwd_r['source_rr'])
rad, tan1, tan2 = conpy.forward._make_radial_coord_system(fwd_r['source_rr'], center)
fig2 = conpy.forward._plot_coord_system(fwd_r['source_rr'], rad, tan1, tan2, scale=0.003, n_ori=2)
fig2.scene.background = (1, 1, 1)  # white
mlab.view(75.115, 47.534, 0.311, [0.00068598, 0.01360262, 0.03581326])
mlab.savefig('../paper/figures/forward2.png', magnification=4, figure=fig2)
