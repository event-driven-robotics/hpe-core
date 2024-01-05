search_dir="/home/schiavazza/data/hpe/vicon_dataset/processed/simon"
echo "${search_dir}/*/"
for base_dir in ${search_dir}/*/   # list directories in the form "/tmp/dirname/"
do
    base_dir=${base_dir%*/}      # remove the trailing "/"
    echo "processing ${dir##*/}..."    # print everything after the final "/"

    atis_s="atis-s"
    atis_d="atis-d"
    atis_s_frames="atis_s_frames"
    atis_d_frames="atis_d_frames"

    input_static="${base_dir}/${atis_s}/"
    input_dynamic="${base_dir}/${atis_d}/"

    frames_static="${base_dir}/${atis_s_frames}/"
    frames_dynamic="${base_dir}/${atis_d_frames}/"

    if [ ! -f ${frames_static}/labeled_points.yml ]; then
        echo "File not found!"
        echo "Labeling ${input_static}"
        python3 ../label_dvs.py --dvs_path ${input_static} --frames_path ${frames_static} --output_path ${frames_static}/labeled_points.yml --calib_labels ../config/calib_labels_slow.yml
    fi

    
    if [ ! -f ${frames_dynamic}/labeled_points.yml ]; then
        echo "File not found!"
        echo "Labeling ${input_dynamic}"
        python3 ../label_dvs.py --dvs_path ${input_dynamic} --frames_path ${frames_dynamic} --output_path ${frames_dynamic}/labeled_points.yml --calib_labels ../config/calib_labels_slow.yml
    fi

done