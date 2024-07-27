Background
==========
The 'Destrieux' cortical atlas is based on a parcellation scheme
that first divided the cortex into gyral and sulcal regions,
the limit between both being given by the curvature value of the surface.
A gyrus only includes the cortex visible on the pial view,
the hidden cortex (banks of sulci) are marked sulcus.
The result is a complete labeling of cortical sulci and gyri.

Files in Folder (Excluding README)
==================================
1) destrieux2009_rois.nii.gz
2) destrieux2009_rois_labels.csv

Optionally, if the lateralized argument is set to True, the archive will also contain:

3) destrieux2009_rois_lateralized.nii.gz
4) destrieux2009_rois_labels_lateralized.csv

Descriptions of files
=====================
1) destrieux2009_rois.nii.gz is a volume consisting of 75 rois projected into a
   2mm isotropic MNI152 space.
2) destrieux2009_rois_labels.csv is the file that contains the ROI labels.
3) destrieux2009_rois_lateralized.nii.gz is a volume consisting of 150 rois,
   75 for each hemisphere, projected into a 2mm isotropic MNI152 space.
4) destrieux2009_rois_labels_lateralized.csv is the file that contains the ROI
   labels for the lateralized volume.

References
==========
Fischl, Bruce, et al. "Automatically parcellating the human cerebral cortex."
Cerebral cortex 14.1 (2004): 11-22.

Destrieux, C., et al. "A sulcal depth-based anatomical parcellation
of the cerebral cortex." NeuroImage 47 (2009): S151.
