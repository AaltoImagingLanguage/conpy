import conpy, mne  # Import required Python modules
con = conpy.read_connectivity('contrast-con.h5')  # Load connectivity object
l = mne.read_labels_from_annot('fsaverage', 'aparc')  # Get parcels from atlas
del l[-1]  # Drop the last parcel (unknown-lh)
# Parcellate the connectivity object and correct for the degree bias
con_parc = con.parcellate(l, summary='degree', weight_by_degree=True)  
con_parc.plot()  # Plot a circle diagram showing connectivity between parcels
# Plot a vertex-wise degree map and connect for the degree bias
brain = con.make_stc('degree', weight_by_degree=True).plot(hemi='split')
brain.add_annotation('aparc')  # Draw the 'aparc' atlas on the degree-map
