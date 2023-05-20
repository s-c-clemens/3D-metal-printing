TITLE       = "PIPAD Printing Setup"
DESCRIPTION = """
All vendor lenses could be used just like any other elements. Remember to 
check backFocalLength() and effectiveFocalLengths() to understand that the focal
point is not "f_e" after the lens but rather "BFL" after the lens.
"""

from raytracing import *

class UMPlanFL10X(Objective):
    """ Olympus 10x immersion objective
    .. csv-table::
        :header: Parameter, value
        "Magnification", "10x"
        "focusToFocusLength", "10"
        "backAperture", "7"
        "Numerical Aperture (NA)", "0.30"
        "Cover Glass Thickness (mm):", "0.00"
        "Diameter (mm)", "24.00"
        "Field Number (mm)", "26.5"
        "Length (mm)", "34.90"
        "Working Distance (mm)", "10"
    Notes
    -----
    Immersion not considered at this point.
    More info: https://www.edmundoptics.com/p/olympus-uplfln-10x-objective/29227
    """

    def __init__(self):
        super(UMPlanFL10X, self).__init__(f=18.00,
                                         NA=0.3,
                                         focusToFocusLength=10,
                                         backAperture=13.6,
                                         workingDistance=10,
                                         magnification=10,
                                         fieldNumber=26.5,
                                         label='UMPlanFL10X',
                                         url="https://www.edmundoptics.com/p/olympus-uplfln-10x-objective/29227")

class AC254_030_AB_ML(AchromatDoubletLens):
    """ AC254-030-A
    .. csv-table::
        :header: Parameter, value
        "fa", "30.0"
        "fb", "21.22"
        "R1", "20.00"
        "R2", "-17.4"
        "R3", "-93.1"
        "tc1", "12.0"
        "tc2", "3.0"
        "te", "9.5"
        "n1", "0.3650"
        "n2", "0.4050"
        "diameter", "25.4"
        "Design Wavelengths", "486.1 nm, 587.6 nm, and 656.3 nm"
        "Operating Temperature", "-40 °C to 85 °C"
    Notes
    -----
    More info: https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=12804&pn=AC254-030-AB-ML
    """

    def __init__(self, wavelength=None):
        super(AC254_030_AB_ML, self).__init__(fa=30.0, fb=21.22, R1=20.00, R2=-17.4, R3=-93.1,
                                    tc1=12.0, tc2=3.0, te=9.5, n1=0.365, n2=0.405, mat1=S_BAH11, mat2=S_TIH6, diameter=25.4,
                                    url='https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=12804&pn=AC254-030-AB-ML',
                                    label="AC254-030-AB-ML", wavelength=wavelength, wavelengthRef=0.5876)


class S_BAH11(Material):
    """ All data from https://refractiveindex.info/tmp/data/glass/ohara/S-BAH11.html """
    @classmethod
    def n(cls, wavelength):
        if wavelength > 10 or wavelength < 0.01:
            raise ValueError("Wavelength must be in microns")
        x = wavelength
        n=(1+1.5713886/(1-0.00910807936/x**2)+0.147869313/(1-0.0402401684/x**2)+1.28092846/(1-130.399367/x**2))**.5
        return n

    @classmethod
    def abbeNumber(cls):
        return 48.32


class S_TIH6(Material):
    """ All data from https://refractiveindex.info/tmp/data/glass/ohara/S-TIH6.html """
    @classmethod
    def n(cls, wavelength):
        if wavelength > 10 or wavelength < 0.01:
            raise ValueError("Wavelength must be in microns")
        x = wavelength
        n=(1+1.77227611/(1-0.0131182633/x**2)+0.34569125/(1-0.0614479619/x**2)+2.40788501/(1-200.753254/x**2))**.5
        return n

    @classmethod
    def abbeNumber(cls):
        return 25.42


def exampleCode(comments=None):

    # FIXME: Some ray bouncing going on in the objective
    path = ImagingPath()
    path.label = TITLE
    path.append(Space(d=15))
    path.append(AC254_030_AB_ML())
    path.append(Space(20))
    path.append(UMPlanFL10X())
    path.append(Space(15))
    path.displayWithObject(diameter=10, fanAngle=0.05, comments=comments)

if __name__ == "__main__":
    exampleCode()
