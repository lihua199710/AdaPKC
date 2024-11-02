#zxy@2022060603
import numpy as np
import math as mt
import os
import scipy.stats as st

# configs
GUARD_CELLS = 6
BG_CELLS = 2
# filter threshold of CFAR
ALPHA = 4.2
# the width and height of CFAR unit
CFAR_UNITS = 1 + (GUARD_CELLS * 2) + (BG_CELLS * 2)
HALF_CFAR_UNITS = int(CFAR_UNITS / 2) + 1
X_SHAPE = 256
Y_SHAPE = 256
# path
OUTPUT_IMG_DIR = "./test_out/"
INPUT_IMG_DIR = "./test_input/"
root = './range_angle_numpy/'


# 2D-CA-CFAR
def gen_ra_proposals_cfar(ra_matrix):  # cfar
    """
    Parameters
    ----------
    ra_matrix

    Returns
    -------
    ra_proposals: coordinate of resevered ra points.
    zero_dop_idx:
    estimateImg_thf: resevered ra points in the ra_matrix.
    """
    inputImg = ra_matrix
    estimateImg = np.zeros((inputImg.shape[0], inputImg.shape[1]), np.uint8)

    # 1. reserve ra points which is obvious using CFAR
    # search
    # (i, j) is the top left corner of CFAR_UNITS
    for i in range(inputImg.shape[0] - CFAR_UNITS):
        center_cell_x = i + BG_CELLS + GUARD_CELLS
        for j in range(inputImg.shape[1] - CFAR_UNITS):
            center_cell_y = j + BG_CELLS + GUARD_CELLS
            # calculate the average energy of background.
            average = 0
            for k in range(CFAR_UNITS):
                for l in range(CFAR_UNITS):
                    # (k, l) in guard area or the center.
                    if (k >= BG_CELLS) and (k < (CFAR_UNITS - BG_CELLS)) and (l >= BG_CELLS) and (
                            l < (CFAR_UNITS - BG_CELLS)):
                        continue
                    average += inputImg[i + k, j + l]
            average /= (CFAR_UNITS * CFAR_UNITS) - (((GUARD_CELLS * 2) + 1) * ((GUARD_CELLS * 2) + 1))

            if inputImg[center_cell_x, center_cell_y] > (average * ALPHA):
                estimateImg[center_cell_x, center_cell_y] = inputImg[center_cell_x,center_cell_y]

    ra_proposals = []
    #rd_filtering
    # 2. keep the top 50 points in the aspect of energy.
    estimateImg_thf, zero_dop_idx = ra_threshold_filtering_cfar(estimateImg,top_val0=50)
    for i in range(estimateImg_thf.shape[0]):
        for j in range(estimateImg_thf.shape[1]):
            if estimateImg_thf[i][j]>0:
                ra_proposals.append([i,j])

    return ra_proposals,zero_dop_idx,estimateImg_thf


def ra_threshold_filtering_cfar(ra_matrix, top_val0=100):  # Filter the top 100
    rng_dim = ra_matrix.shape[0]
    dop_dim = ra_matrix.shape[1]
    zero_dop_idx = mt.ceil(dop_dim / 2)
    rd_vec = np.sort(ra_matrix.reshape(rng_dim * dop_dim))
    vth0 = rd_vec[-1 * top_val0]
    ra_matrix_thf = np.zeros([rng_dim, dop_dim])
    for i in range(rng_dim):
        for j in range(dop_dim):
            if ra_matrix[i][j] > vth0:
                ra_matrix_thf[i][j] = ra_matrix[i][j]
    return ra_matrix_thf, zero_dop_idx


def compute_match_score_zxy(a,b,ra_matrix_thf):
    if np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) == 0:
        score = 2
    else:
        score = 1 / np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    t_score = score * ra_matrix_thf[a[0]][a[1]]
    return t_score


def compute_range_angle(ra_coord, zero_dop_idx):
    RANGE_RES = 0.1953125
    ANGLE_RES = 0.01227184630308513
    ra_prop_range = (X_SHAPE-ra_coord[0]) * RANGE_RES
    ra_prop_angle = np.rad2deg((ra_coord[1] - zero_dop_idx) * ANGLE_RES)
    return ra_prop_range, ra_prop_angle

def ra_coord_calibration(ra_proposals,
                         ra_matrix_thf,
                         zero_dop_idx,
                         img_range_coord,
                         img_angle_coord):
    img_coord = [img_range_coord, img_angle_coord]
    match_scores = []
    if ra_proposals:
        for coord in ra_proposals:
            match_scores.append(compute_match_score_zxy(coord, img_coord, ra_matrix_thf))
        matched_index = match_scores.index(max(match_scores))
        calibrated_coord = ra_proposals[matched_index]
    else:
        calibrated_coord = img_coord
    _range, _angle = compute_range_angle(calibrated_coord,
                                         zero_dop_idx)
    calibrated_range = _range
    calibrated_angle = _angle

    # print('prop:', rd_proposals)
    # print('clb:', calibrated_coord)
    # print('img:', img_range_coord, img_velocity_coord)
    return calibrated_coord, calibrated_range, calibrated_angle


def rd_gaussian_calibration(rd_proposals,
                         rd_matrix_thf,
                         zero_dop_idx,
                         img_range_coord,
                         img_velocity_coord):

    img_map = gaussian_distribution_img(img_range_coord,img_velocity_coord)
    radar_map = gaussian_distribution_radar(rd_proposals)
    map = img_map * radar_map

    return map


def gaussian_distribution_img(img_range_coord,img_velocity_coord):
    mean = np.array([img_range_coord, img_velocity_coord])
    cov = np.array([[10, 0], [0, 2]])
    x, y = np.mgrid[0:256, 0:64]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv = st.multivariate_normal(mean, cov)
    map = rv.pdf(pos)

    return map


def gaussian_distribution_radar(rd_proposals):
    c_map = np.zeros((len(rd_proposals),256,64))
    i = 0
    for coord in rd_proposals:
        mean = np.array([coord[0], coord[1]])
        cov = np.array([[0.5, 0], [0, 1]])
        x, y = np.mgrid[0:256, 0:64]
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        rv = st.multivariate_normal(mean, cov)
        c_map[i] = rv.pdf(pos)
        i = i + 1
    map = maximized_gaussian(c_map)
    return map


def maximized_gaussian(c_map):
    map = np.zeros((256, 64))
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            map[i][j] = max(c_map[k][i][j] for k in range(c_map.shape[0]))

    # Normalize
    sum = np.sum(map)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            map[i][j] = map[i][j] / sum
    return map