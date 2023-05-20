Instructions

To run the code, run Main.py. Main.py pulls from the other python files (data.py, Functions.py) and prints the results. When you run Main.py, it will print you a list of wavelengths that we have tested for and prompt you to enter one of the wavelengths to get the results. The code will take a few minutes to print the results, and it will take longer if the 3D plot is run. Under ## Variables ## you may choose to change L or L2, but leave the rest untouched as they are imports.
There are two other python scipts in the folder as well: Raytracing_Test.py and wavelengths_focal.py. Raytracing_Test.py is an example of ray tracing using the raytracing python library where it shows the same Thorlabs lens in our project. Wavelengths_focal.py is just a simple plotting script to create a plot of wavelength vs focal length. Neither of these are crucial to our main code, but you may look at them.

Warning : When running the code, it has been seen that VSCode will not recognize the input picture even though it shares the same directory. This issue has only been see in VSCode and other Python IDEs have worked fine.

Note : 3D Plotting is commented out by default as it takes a while to run, and it does not really aid in understanding more than the 2D plots.

Note : The 2D plots of the XY images is commented out by default as the images produced are just overlapping solids of each lens surface. The design printing plots show more specific detail so this renders the 2D XY plots obsolete.

Note : Ensure that 'blank.bmp' is in the same folder as the python scripts.