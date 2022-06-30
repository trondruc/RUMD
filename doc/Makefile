all:
	make -C tutorial all
	make -C usermanual

# Build doxygen generated files from toplevel directory
doxydoc:
	mkdir -p doxydoc
	(cd .. ; doxygen doc/Doxyfile )
	@echo "doxydoc available at file:///"${PWD}"/doxydoc/html/index.html"

clean:
	make -C tutorial clean
	make -C usermanual clean
	rm -rf doxydoc

.PHONY: all clean doxydoc
