import conpy, mne  # Import required Python modules

# Define source space on average brain, morph to subject
src_avg = mne.setup_source_space("fsaverage", spacing="ico4")
src_sub = mne.morph_source_spaces(src_avg, subject="sub002")

# Discard deep sources
info = mne.io.read_info("sub002-epo.fif")  # Read information about the sensors
verts = conpy.select_vertices_in_sensor_range(src_sub, dist=0.07, info=info)
src_sub = conpy.restrict_src_to_vertices(src_sub, verts)

# Create a one-layer BEM model
bem_model = mne.make_bem_model("sub002", ico=4, conductivity=(0.3,))
bem = mne.make_bem_solution(bem_model)

# Make the forward model
trans = "sub002-trans.fif"  # File containing the MRI<->Head transformation
fwd = mne.make_forward_solution(info, trans, src_sub, bem, meg=True, eeg=False)

# Only retain orientations tangential to a sphere approximation of the head
fwd = conpy.forward_to_tangential(fwd)
