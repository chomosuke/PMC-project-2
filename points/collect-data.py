file_names = ["slurm-1-core.out", "slurm-64-nodes.out"]

content = []
for file_name in file_names:
    with open("../" + file_name) as f:
        content += f.read().split("\n")

core = 0
c_d = {}
keys = ["generate", "center", "k-means", "localize", "simulation"]
for l in content:
    s = l.split(" ")
    if len(s) == 2 and s[1] == "cores":
        if core in c_d:
            for k in keys:
                c_d[core][k] /= core
        core = int(s[0])
        c_d[core] = {}
        for k in keys:
            c_d[core][k] = 0
    elif len(s) > 2:
        if s[2] == "Points:":
            c_d[core]["generate"] += float(s[6])
        elif s[2] == "generate":
            c_d[core]["center"] += float(s[6])
        elif s[2] == "k-means":
            c_d[core]["k-means"] += float(s[5])
        elif s[2] == "localizing":
            c_d[core]["localize"] += float(s[5])
        elif s[2] == "simulation":
            c_d[core]["simulation"] += float(s[5])

for k in keys:
    c_d[core][k] /= core

for k in keys:
    print(k)
    for core in c_d:
        print(c_d[core][k])
