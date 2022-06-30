

def init_mol_simplelin(nmol, nuau):

    file=open("mol.top", "w")

    file.write("[ bonds ]\n");
    file.write("; mi ai aj btype\n");

    atomc=0
    for n in range(0,nmol):
        for m in range(0,nuau-1):
            string = "%d %d %d 0\n" % (n, atomc, atomc+1)
            file.write(string)
            atomc = atomc + 1
        atomc = atomc + 1 
            
    file.write("\n")
    file.close()

    

    
