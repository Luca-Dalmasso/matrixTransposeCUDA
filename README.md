# matrix transpose algorithm CUDA-C
<br>
Simple example of a typical work flow to solve and optimize algorithm running on GPU<br>
with the support of command line **nvprof** profiling tool from Nvidia.<br>
*NOTE*: this project has been tested on *Nvidia's Tegra X1 GPU* (Maxwell architecture)<br>
		**you might have different behaviour if you run the code on a different GPU**<br>
*NOTE*: it is *required* to have installed *Nvidia CUDA Toolkit* on your machine<br>
*NOTE*: the makefile also supports the doxygen documentation, it is required to have<br>
		doxygen on your machine, *Doxyfile* is configured to automatically generate html<br>
		and LaTex documentation. LaTex documentation can be convered in pdf using another<br>
		*Makefile in docs/latex/* folder, but this requires to install additional tools.<br>
		*See the documentation section*
<br>

## Documentation
The makefile does integrate Doxygen tool which is is the de facto standard used for generating documentation <br>
from annotated C, C++, C#, python, others.. source code. (additional informations https://www.doxygen.nl/index.html).<br>

HTML documentation only requires Doxygen
LaTex (pdf) documentation requires additional tools (not mandatory if you do not need to generate the reference manual in pdf)

### Install Doxygen

*Install Doxygen (~100MB)*
'''
sudo apt-get install doxygen
'''

### Install Pdflatex

*Install TexLive base (~150 MB)*
'''
sudo apt-get install texlive-latex-base
'''

*Install additional fonts (~1300MB)*
'''
sudo apt-get install texlive-fonts-recommended
sudo apt-get install texlive-fonts-extra
'''



