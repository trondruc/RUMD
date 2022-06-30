%typemap(in) std::vector<unsigned> {
  
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
    return NULL;
  }

  int numTypes = PyObject_Length($input);

  for (int i =0; i < numTypes; i++) {
    PyObject *o = PySequence_GetItem($input,i);

    if (!PyInt_Check(o)) {
         Py_XDECREF(o);
         PyErr_SetString(PyExc_ValueError,"Expecting a sequence of ints");
         return NULL;
      }

    $1.push_back(PyInt_AsLong(o));
    Py_DECREF(o);
  }
 }


%typemap(in) std::vector<double> {
  
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
    return NULL;
  }

  int numTypes = PyObject_Length($input);

  for (int i =0; i < numTypes; i++) {
    PyObject *o = PySequence_GetItem($input,i);

    if (!PyFloat_Check(o)) {
         Py_XDECREF(o);
         PyErr_SetString(PyExc_ValueError,"Expecting a sequence of floats");
         return NULL;
      }

    $1.push_back( PyFloat_AsDouble(o));
    Py_DECREF(o);
  }
 }


%typemap(in) std::vector<float> {
  
  if (!PySequence_Check($input)) {
    PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
    return NULL;
  }

  int numTypes = PyObject_Length($input);

  for (int i =0; i < numTypes; i++) {
    PyObject *o = PySequence_GetItem($input,i);

    if (!PyFloat_Check(o)) {
         Py_XDECREF(o);
         PyErr_SetString(PyExc_ValueError,"Expecting a sequence of floats");
         return NULL;
      }

    $1.push_back( PyFloat_AsDouble(o));
    Py_DECREF(o);
  }
 }

