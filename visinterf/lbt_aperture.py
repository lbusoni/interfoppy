import poppy
from astropy.io import fits
from poppy.poppy_core import PlaneType
import os
from interfoppy.package_data import data_root_dir
from scipy.ndimage import rotate

PUPIL_MASK_480 = 'EELT480pp0.0813spider.fits'
PUPIL_MASK_480_PHASE_B = 'EELT480pp0.0813spiderRoundObs.fits'
PUPIL_MASK_480_PHASE_C = 'EELT480pp0.0813spiderRoundObsCentered.fits'
PUPIL_MASK_512 = 'EELT512pp0.0762nogapRoundObs.fits'
PUPIL_MASK_480_PHASE_C_SPIDER23 = 'EELT480pp0.0803m_obs0.283_spider2023.fits'
PUPIL_MASK_480_PHASE_C_SPIDER23_HIRES = 'EELT480pp0.0803m_obs0.283_spider2023.fits'
PUPIL_MASK_DEFAULT = PUPIL_MASK_480_PHASE_C_SPIDER23
PUPIL_MASK_NONE = 'None'


def restore_lbt_pupil_mask(pupil_mask_tag):

    fname = os.path.join(data_root_dir(),
                         'pupilstop',
                         pupil_mask_tag)
    mask = fits.getdata(fname)
    maskb = (mask == False)
    maski = maskb.astype(int)
    # pixel_pitch must be read from params.txt MAIN PIXEL_PITCH
    # rotation: such that a spider is pointing N (with pixel 0,0 at the SW corner)
    if pupil_mask_tag == PUPIL_MASK_480:
        pixel_pitch = 0.08215
        rotation = 0
    elif pupil_mask_tag == PUPIL_MASK_480_PHASE_B:
        pixel_pitch = 0.081249997
        rotation = 0
    elif pupil_mask_tag == PUPIL_MASK_480_PHASE_C:
        pixel_pitch = 0.080208331
        rotation = 0
    elif pupil_mask_tag == PUPIL_MASK_512:
        pixel_pitch = 0.076171875
        rotation = 15
    elif pupil_mask_tag == PUPIL_MASK_480_PHASE_C_SPIDER23:
        pixel_pitch = 0.080208331
        rotation = 0
    elif pupil_mask_tag == PUPIL_MASK_480_PHASE_C_SPIDER23_HIRES:
        pixel_pitch = 0.081249997
        rotation = 0
    else:
        raise ValueError('Unknown pupilstop %s' % pupil_mask_tag)
    hdr = fits.Header()
    hdr['PIXELSCL'] = pixel_pitch
    hdr['ROTATION'] = rotation

    mask = rotate(maski, rotation, reshape=False,
                  cval=1, mode='constant')
    hdu = fits.PrimaryHDU(data=mask, header=hdr)
    return hdu


class LBTAperture(poppy.FITSOpticalElement):

    def __init__(self, pupil_mask_tag, **kwargs):
        hdumask = restore_lbt_pupil_mask(pupil_mask_tag)
        hdumask.data = LBTAperture._invert_int_mask(hdumask.data)
        self.pupil_mask_tag = pupil_mask_tag
        poppy.FITSOpticalElement.__init__(self,
                                          transmission=fits.HDUList([hdumask]),
                                          planetype=PlaneType.pupil, **kwargs)

    @staticmethod
    def _invert_int_mask(mask):
        return -mask + 1
