
// This pair of typemaps allows you to use std::string& args

%typemap(in) const std::string & {
  if (PyString_Check($input)) {
    $1 = new std::string(PyString_AsString($input));
  }
  // for Python 3
  else if (PyUnicode_Check($input)) {
    PyObject* coded_string = PyUnicode_AsEncodedString($input, "utf-8", "strict");
    char* byte_string = PyBytes_AsString(coded_string);
    if(byte_string)
      $1 = new std::string(byte_string);
    else {
      SWIG_exception(SWIG_TypeError, "Failed to convert unicode string to bytes");
	}
    Py_DECREF(coded_string);
    }
  else { 
    SWIG_exception(SWIG_TypeError, "string expected");
  }
 }

%typemap(typecheck, precedence=SWIG_TYPECHECK_STRING) const std::string &{
  $1 = PyString_Check($input) ? 1 : 0;
 }

// This typemap cleans up the new() above.
%typemap(freearg) const std::string & {
	delete ($1);
 }


// This typemap allows you to return a std::string                              
%typemap(out) std::string, const std::string {
  $result = PyString_FromString($1.c_str());
 }
