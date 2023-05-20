import matplotlib.pyplot as plt

f = [20.18, 20.17, 20.15]
others = [488, 707, 1064]
wavelengths = [365, 405, 436, 488, 707, 1064]
plt.plot(365, 19.36, 'bo')
plt.plot(405, 19.90, 'go')
plt.plot(436, 20.19, 'ro')
plt.plot(others, f, 'ko')
plt.title(f"Focal Point of Various Wavelengths")
plt.xlabel("$\lambda$ [nm]")
plt.ylabel("f [mm]")
plt.ylim(18.5, 20.5)
plt.xticks(wavelengths, rotation=45)
plt.savefig(f"Focal_Points.png")
plt.tight_layout()

plt.show()
