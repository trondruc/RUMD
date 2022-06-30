UserManual.pdf: UserManual.tex pot_list.tex
	pdflatex $<
	pdflatex $<
	pdflatex $<

clean:
	-rm -f UserManual.log UserManual.toc UserManual.aux UserManual.pdf
