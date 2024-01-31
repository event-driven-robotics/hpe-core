search_dir="/home/schiavazza/data/hpe/vicon_dataset/processed/roberta"
echo "${search_dir}/*/"

status_file="./status_labeled.txt"

for base_dir in ${search_dir}/*/   # list directories in the form "/tmp/dirname/"
do
    base_dir=${base_dir%*/}      # remove the trailing "/"
    echo "processing ${base_dir##*/}..."    # print everything after the final "/"

    atis_s="atis-s"
    atis_d="atis-d"
    atis_s_frames="atis_s_frames"
    atis_d_frames="atis_d_frames"

    input_static="${base_dir}/${atis_s}/"
    input_dynamic="${base_dir}/${atis_d}/"

    frames_static="${base_dir}/${atis_s_frames}/"
    frames_dynamic="${base_dir}/${atis_d_frames}/"

    seq_name=$(basename "${base_dir}")
    c3d_base_dir=$(dirname "${base_dir}")
    c3d_file="$c3d_base_dir/$seq_name.c3d"

    echo "${seq_name}: " >> $status_file

    if [ -f ${frames_static}/labeled_points.yml ]; then
        echo -e "\tstatic: LABELED" >> $status_file
    else
        echo -e "\tstatic: NOT LABELED" >> $status_file
    fi

    
    if [ -f ${frames_dynamic}/labeled_points.yml ]; then
        echo -e "\tdynamic: LABELED" >> $status_file
    else
        echo -e "\tdynamic: NOT LABELED" >> $status_file
    fi

done