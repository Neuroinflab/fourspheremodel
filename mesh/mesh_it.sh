# High resolution mesh by default
if [ $# -eq 0 ]
  then
    set -- sphere_4
fi

for NAME in "$@"
do
  gmsh -3 -optimize_netgen "${NAME}.geo"
  dolfin-convert "${NAME}.msh" "${NAME}.xml"
done
