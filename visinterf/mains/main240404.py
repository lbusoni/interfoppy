

import scipy
import numpy as np
from arte.utils.paste import paste


class LbtInterferometry:

    def __init__(self):
        pass

    @staticmethod
    def from_soul_residual(residual_sav_file, phase_matrix_sav_file):
        li = LbtInterferometry()
        wf_cube = li.restore_residual_wavefront(residual_sav_file)
        li.restore_sx_dx_residual_wavefront(wf_cube)
        li.make_interferometric_wavefront()
        mask = li.restore_mask(phase_matrix_sav_file)
        li.make_interferometric_mask(mask)
        return li

    def restore_residual_wavefront(self, residual_fname):
        res = scipy.io.readsav(residual_fname)
        wf_cube = res['res_opd_cube']
        return wf_cube

    def restore_sx_dx_residual_wavefront(self, wf_cube):
        self._single_pupil_shape = wf_cube[0].shape
        self._single_pupil_diam_px = wf_cube[0].shape[0]
        self._n_iter = wf_cube.shape[0] // 2
        self._wf_sx = wf_cube[0:self._n_iter, :, :]
        self._wf_dx = wf_cube[-self._n_iter:, :, :]

    def restore_mask(self, phase_matrix_fname):
        pm = scipy.io.readsav(phase_matrix_fname)
        mask = np.zeros((pm['dpix'], pm['dpix']))
        mask.flat[pm['idx_mask']] = 1
        return mask

    def make_interferometric_wavefront(self):
        self._pupil_diam_px = round(self._single_pupil_diam_px/8.4*23)
        self._wfi = np.zeros(
            (self._n_iter, self._pupil_diam_px, self._pupil_diam_px))
        for i in range(self._n_iter):
            paste(self._wfi[i], self._wf_sx[i],
                  ((self._pupil_diam_px-self._single_pupil_diam_px)//2, 0))
            paste(self._wfi[i], self._wf_dx[i], ((self._pupil_diam_px-self._single_pupil_diam_px) //
                  2, self._pupil_diam_px-self._single_pupil_diam_px))

    def make_interferometric_mask(self, single_pupil_mask):
        self._maski = np.zeros((self._pupil_diam_px, self._pupil_diam_px))
        paste(self._maski, single_pupil_mask,
              ((self._pupil_diam_px-self._single_pupil_diam_px)//2, 0))
        paste(self._maski, single_pupil_mask, ((self._pupil_diam_px-self._single_pupil_diam_px) //
              2, self._pupil_diam_px-self._single_pupil_diam_px))

    def _roi(self):
        fov_px = 60
        center = (self._padding_factor * self._pupil_diam_px) // 2
        return (center-fov_px//2, center+fov_px//2)

    def cut_roi(self, psf):
        roi = self._roi()
        return psf[roi[0]:roi[1], roi[0]:roi[1]]

    def short_exposure_psf(self, idx, wl):
        self._padding_factor = 5
        ef = self._maski*np.exp(1j*self._wfi[idx]*2*np.pi/wl)
        efp = np.pad(ef, self._pupil_diam_px*(self._padding_factor-1)//2)
        psf = np.abs(np.fft.fftshift(np.fft.fft2(efp)))**2
        return psf

    def compute_psf_series(self, wl):
        res = []
        for i in range(10):
            res.append(self.cut_roi(self.short_exposure_psf(i, wl)))
        self._psf_series = np.array(res)

    def compute_psf_dl(self):
        self._psf_dl = self.cut_roi(self.short_exposure_psf(0, 1e12))

    def single_pupil_strehl(self):
        np.exp(-(dd[100:400, 0:200].std() / 750e-9 * 2 * np.pi)**2)

    def masked_wf(self):
        dd = np.ma.array(wfi[-1], mask=1-maski)


def main():
    residual_fname = '/Users/lbusoni/Downloads/20230507_041220_res_OPD.sav'
    phase_matrix_fname = '/Users/lbusoni/Downloads/phase_matrix.sav'
    li = LbtInterferometry.from_soul_residual(
        residual_fname, phase_matrix_fname)
    return li
