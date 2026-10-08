"""Face ROIs shared by the pipeline stages: st1 builds the masks, st2-st4 pick ROIs by name.

ROIs (as in Cazettes et al. 2025): horizontal bands of the face, full face width.
name -> (top landmark, bottom landmark, cut out the eyes). Each band runs from the top
landmark's row down to (not including) the bottom landmark's row, in the median face;
None = no limit on that side. Every ROI is limited to the face oval, so the band's width
is the face's width at each row. Landmark numbers are MediaPipe FaceMesh indices.
"""

# Upper/lower split at mid-nose (195), below the eyes and their margin, so the eyes are in
# upper_face as in Cazettes et al. (the nose bridge, 168, cuts through the top of the eyes).
FOREHEAD_TOP, MID_NOSE, CHIN = 10, 195, 152
MID_FOREHEAD = 151
ABOVE_BROWS = 9                   # forehead, just above the brows
BROWS_MID = 8                     # between the brows
NOSE_BRIDGE = 197                 # nose bridge, just below the eyes

ROIS = {
    "whole_face":               (None, None, False),
    "upper_face":               (FOREHEAD_TOP, MID_NOSE, False),
    "lower_face":               (MID_NOSE, CHIN, False),
    "upper_face_no_eyes":       (FOREHEAD_TOP, MID_NOSE, True),
    # Experimental: upper_face with the top edge lowered step by step
    "mid_forehead_to_mid_nose": (MID_FOREHEAD, MID_NOSE, False),
    "brows_to_mid_nose":        (ABOVE_BROWS, MID_NOSE, False),
    "eye_band":                 (BROWS_MID, NOSE_BRIDGE, False),
}

# The ROIs st2 and st4 use when --rois isn't given (the main analysis). The experimental
# ones are left out so they don't add features to st4's combined model by default.
DEFAULT_ROIS = ["whole_face", "upper_face", "lower_face", "upper_face_no_eyes"]
