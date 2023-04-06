![image](https://github.com/patternizer/guy-callendar/blob/main/guy-callendar-gmst.png)

# guy-callendar

Python code to reverse engineer Guy Callendar's 1938 GMST using GloSAT stations. 

## Contents

* `guy-callendar.py` - python code to extract tropics and temperate zones temperature series and compute GMST

The first step is to clone the latest guy-callendar code and step into the check out directory: 

    $ git clone https://github.com/patternizer/guy-callendar.git
    $ cd guy-callendar

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested in a conda virtual environment running a 64-bit version of Python 3.8+.

guy-callendar scripts cannot be run from sources directly but require the GloSAT station archive.

Run with:

    $ python guy-callendar.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)

