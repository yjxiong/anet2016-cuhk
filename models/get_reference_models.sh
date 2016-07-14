# bn flow model was trained with cuDNN v4, we will use the weights upgraded to cuDNN v5
wget -O models/bn_inception_anet_2016_temporal.caffemodel.v5 http://personal.ie.cuhk.edu.hk/~xy012/others/models/bn_inception_anet_2016_temporal.caffemodel.v5

wget -O models/bn_inception_anet_2016_temporal_deploy.prototxt http://personal.ie.cuhk.edu.hk/~xy012/others/models/bn_inception_anet_2016_temporal_deploy.prototxt
wget -O models/resnet200_anet_2016.caffemodel http://personal.ie.cuhk.edu.hk/~xy012/others/models/resnet200_anet_2016.caffemodel
wget -O models/resnet200_anet_2016_deploy.prototxt http://personal.ie.cuhk.edu.hk/~xy012/others/models/resnet200_anet_2016_deploy.prototxt