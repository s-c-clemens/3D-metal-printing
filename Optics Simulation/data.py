# DLP Lens Correction Data
# Scott Clemens, Eric Everett
# PHY 432 Final Project


# refractive indices at 365nm, 405nm, 436nm
# First and last medium is air
# from https://refractiveindex.info/?shelf=other&book=air&page=Ciddor
n_air = {'365' : 1.00028495,
         '405' : 1.00028250,
         '436' : 1.00028108,
         '488' : 1.00027930,
         '707' : 1.00027580,
         '1064' : 1.00027398
         }

# Lens 1 is S-BAH11
# from https://refractiveindex.info/?shelf=glass&book=OHARA-BAH&page=S-BAH11
n_lens1 = {'365' : 1.7021,
         '405' : 1.6906,
         '436' : 1.6841,
         '488' : 1.6761,
         '707' : 1.6603,
         '1064' : 1.6511
         }

# Lens 2 is S-TIH6
# from https://refractiveindex.info/?shelf=glass&book=OHARA-TIH&page=TIH6
n_lens2 = {'365' : 1.8986,
         '405' : 1.8646,
         '436' : 1.8472,
         '488' : 1.8272,
         '707' : 1.7911,
         '1064' : 1.7734
         }

# Intermediate Lens AC254-030-AB-ML
# all units are in mm
# from https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=12804&pn=AC254-030-AB-ML
lens = {'diameter' : 25.4,
        'fa' : 30.0,
        'fb' : 21.22,
        'WD' : 18.13,
        'R1' : 20.0,
        'R2' : 17.4,
        'R3' : 93.1,
        'tc1' : 12.0,
        'tc2' : 3.0
        }
