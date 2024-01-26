search_dir="/home/schiavazza/data/hpe/vicon_dataset/processed/gaurvi"
echo "${search_dir}/*/"

status_file="./status.txt"

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
        echo "File found!"
        echo "optimising ${input_static}"
        echo -e "\tstatic: " >> $status_file
        python3 ../delay_optimise.py --dvs_path ${input_static} --vicon_path ${c3d_file} --annotated_points ${frames_static}/labeled_points.yml --intrinsic ${c3d_base_dir}/calib-s.txt --extrinsic ${c3d_base_dir}/extrinsic_s.npy --output "vicon_s_delay.txt" --no_camera_markers
        s=$?
        if [ $s -ne 0 ]; then
            echo -e "\t\tFAIL " >> $status_file
        else
            echo -e "\t\tSUCCESS " >> $status_file
        fi
    fi

    
    if [ -f ${frames_dynamic}/labeled_points.yml ]; then
        echo "File found!"
        echo "optimising ${input_dynamic}"
        echo -e "\tdynamic: " >> $status_file
        python3 ../delay_optimise.py --dvs_path ${input_dynamic} --vicon_path ${c3d_file} --annotated_points ${frames_dynamic}/labeled_points.yml --intrinsic ${c3d_base_dir}/calib-d.txt --extrinsic ${c3d_base_dir}/extrinsic_d.npy --output "vicon_d_delay.txt"
        s=$?
        if [ $s -ne 0 ]; then
            echo -e "\t\tFAIL " >> $status_file
        else
            echo -e "\t\tSUCCESS " >> $status_file
        fi
    fi

done