open("T2.txt", 'w').close()

with open("T2.txt", 'w') as f:
    f.write("|")
    for j in range(784):
        f.write("-")
    f.write("|\n")

    for i in range(350):
        f.write("|")
        for j in range(784):
            if i == j: 
                f.write("1")
            else:
                f.write(" ")
        f.write("|\n")
    
    for j in range(786):
        f.write("-")

open("T3.txt", 'w').close()

with open("T3.txt", 'w') as f:
    f.write("|")
    for j in range(784):
        f.write("-")
    f.write("|\n")

    for i in range(7):
        for j in range(7):
            f.write("|")
            for k in range(112 * i): f.write(" ")
            for k in range(4):
                for l in range(4 * j): f.write(" ")
                f.write("1111")
                for l in range(24 - 4 * j): f.write(" ")
            for k in range(672 - 112 * i): f.write(" ")
            f.write("|\n")

    for j in range(786):
        f.write("-")
