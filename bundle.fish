#set pkg_dir (dirname (which python))/../lib/python3.7/site-packages/
#set tf_contrib_sos (string replace -r $pkg_dir\/ '' -- (find $pkg_dir/tensorflow/contrib -name '*.so'))
#pyinstaller inference.py -y -p "./waveglow" \
#    --add-data "*.pt:." --hidden-import sklearn.neighbors.typedefs \
#    --hidden-import sklearn.neighbors.quad_tree --hidden-import sklearn.tree --hidden-import sklearn.tree._utils \
#    (for item in $tf_contrib_sos; echo --add-binary $pkg_dir$item:(dirname $item) | string split " "; end)

pyinstaller inference.py -y -p "./waveglow" \
    --add-data "*.pt:." --hidden-import sklearn.neighbors.typedefs \
    --hidden-import sklearn.neighbors.quad_tree --hidden-import sklearn.tree --hidden-import sklearn.tree._utils
