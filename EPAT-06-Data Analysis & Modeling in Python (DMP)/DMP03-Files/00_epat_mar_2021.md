Executive Program in Algorithmic Trading (QuantInsti)
=====================================================

**Python for Algorithmic Trading Sessions** by

Prof. Dr. Yves J. Hilpisch<br>CEO The Python Quants | The AI Machine 

Online, March 2021

Short Link
----------
http://bit.ly/epat_mar_2021

Slides & Materials
------------------

You find the introduction slides under http://hilpisch.com/epat.pdf

You find the materials about OOP under http://hilpisch.com/py4fi_oop_epat.html

Certificate Programs
--------------------

Platinum Package (both below combined): http://home.tpq.io/certificates

Python for Algorithmic Trading: http://home.tpq.io/certificates/pyalgo

Python for Computational Finance: http://home.tpq.io/certificates/compfin

Python for Algorithmic Trading
---------------------------------
Our new book about Python for Algorithmic Trading (http://py4at.tpq.io).

Sign up under http://py4at.pqp.io to access all the Jupyter Notebooks and execute them on our Quant Platform.


<img src="http://hilpisch.com/pyalgo_cover_color.png" width=300px border=1px>

Artificial Intelligenc in Finance
---------------------------------
Our new book about Artificial Intelligence in Finance (http://aiif.tpq.io).

Sign up under http://aiif.pqp.io to access all the Jupyter Notebooks and execute them on our Quant Platform.


<img src="http://hilpisch.com/aiif_cover_color.png" width=300px border=1px>

Python for Finance (2nd ed.)
----------------------------
Our standard reference book about Python for Finance (http://py4fi.tpq.io).

Sign up under http://py4fi.pqp.io to access all the Jupyter Notebooks and execute them on our Quant Platform.

<img src="http://hilpisch.com/images/py4fi_2nd.png" width=300px border=1px>



Further Resources
-----------------

* http://tpq.io
* http://hilpisch.com
* http://twitter.com/dyjh


Python
------

If you have either Miniconda or Anaconda already installed, there is no need to install anything new.

The code that follows uses Python 3.8. For example, download and install **Miniconda 3.8** from https://docs.conda.io/en/latest/miniconda.html if you do not have `conda` already installed.

In any case, for **Linux/Mac** you should execute the following lines on the shell  to create a new environment with the needed packages:

    conda create -n epat python=3.8
    conda activate epat
    conda install numpy pandas scikit-learn matplotlib statsmodels
    pip install cufflinks
    conda install jupyterlab
    jupyter lab

On **Windows**, execute the following lines on the Anaconda prompt:
    
    conda create -n epat python=3.8
    activate epat
    conda install numpy pandas scikit-learn matplotlib statsmodels
    pip install cufflinks
    pip install win-unicode-console
    set PYTHONIOENCODING=UTF-8
    conda install jupyterlab
    jupyter lab

Read more about the management of environments under

https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

Docker
------

To install **Docker** see https://docs.docker.com/install/.

To run a Ubuntu-based Docker container, execute on the shell (Linux/Mac) the following: 

    docker run -ti -p 9999:9999 -h epat -v $(pwd):/root/live ubuntu:latest /bin/bash

Make sure to adjust the folder to be mounted accordingly.

ZeroMQ
------

The major resource for the `ZeroMQ` distributed messaging package based on sockets is http://zeromq.org/

Cloud
-----
Use this link to get a 10 USD bonus on **[DigitalOcean](https://m.do.co/c/fbe512dd3dac)** when signing up for a new account.


Books & Resources
-----------------

An overview of the **Python Data Model** is found under: [Python Data Model](https://docs.python.org/3/reference/datamodel.html)

Excellent book about everything important in **Python data analysis**: [Python Data Science Handbook, O'Reilly](https://learning.oreilly.com/library/view/python-data-science/9781491912126/)

Standard reference about Deep Learning with the `Keras` package (with `TensorFlow` backend): [Deep Learning with Python, Manning](https://www.manning.com/books/deep-learning-with-python)

Excellent book covering **object-oriented programming in Python**: [Fluent Python, O'Reilly, 2nd ed.](https://learning.oreilly.com/library/view/fluent-python-2nd/9781492056348/)



<img src="http://hilpisch.com/tpq_logo.png" width=250px>
