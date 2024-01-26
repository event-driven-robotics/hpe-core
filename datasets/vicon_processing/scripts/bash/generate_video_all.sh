search_dir="/home/schiavazza/data/hpe/vicon_dataset/processed/zhichao"
echo "${search_dir}/*/"
status_file="./status_video.txt"
subject="P10"

failed=0
success=0

process_sequence () {

    base_dir=$1

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

    if [ -f ${base_dir}/vicon_s_delay.txt ]; then

        echo -e "\tstatic: " >> $status_file
    
        delay_file=${base_dir}/vicon_s_delay.txt
        delay=$(< $delay_file)
        echo "Delay s: ${delay}"
        echo -e "\t\t${delay} " >> $status_file
        output_video="/home//schiavazza/Videos/vicon/zhichao2/${seq_name}_s.mp4"
        echo "Saving video to: ${output_video}"

        if [ -f $output_video ]; then
            echo -e "\t\tALREADY DONE" >> $status_file
        else
            python3 ../generate_video.py --dvs_path ${input_static} --vicon_path ${c3d_file} --intrinsic ${c3d_base_dir}/calib-s.txt --extrinsic ${c3d_base_dir}/extrinsic_s.npy --output_path ${output_video} --labels ../config/labels_joints.yml --vicon_delay ${delay} --no_camera_markers --subject $subject
            s=$?
            if [ $s -ne 0 ]; then
                echo -e "\t\tFAIL " >> $status_file
                failed=$((failed + 1))
            else
                echo -e "\t\tSUCCESS " >> $status_file
                success=$((success + 1))
            fi
        fi
    else
        echo -e "\t\tNO DELAY " >> $status_file
    fi
        

    if [ -f ${base_dir}/vicon_d_delay.txt ]; then
        echo -e "\tdynamic: " >> $status_file
        delay_file=${base_dir}/vicon_d_delay.txt
        delay=$(< $delay_file)
        echo "Delay d: ${delay}"
        echo -e "\t\t${delay} " >> $status_file
        output_video="/home/schiavazza/Videos/vicon/zhichao/${seq_name}_d.mp4"
        echo "Saving video to: ${output_video}"
        if [ -f $output_video ]; then
            echo -e "\t\tALREADY DONE" >> $status_file
        else
            python3 ../generate_video.py --dvs_path ${input_dynamic} --vicon_path ${c3d_file} --intrinsic ${c3d_base_dir}/calib-d.txt --extrinsic ${c3d_base_dir}/extrinsic_d.npy --output_path ${output_video} --labels ../config/labels_joints.yml --vicon_delay ${delay} --subject $subject
            s=$?
            if [ $s -ne 0 ]; then
                echo -e "\t\tFAIL " >> $status_file
                failed=$((failed + 1))
            else
                echo -e "\t\tSUCCESS " >> $status_file
                success=$((success + 1))
            fi
        fi
    else
        echo -e "\t\tNO DELAY " >> $status_file
    fi
}

for seq_dir in ${search_dir}/*/   # list directories in the form "/tmp/dirname/"
do
    seq_dir=${seq_dir%*/}      # remove the trailing "/"
    echo "processing ${seq_dir##*/}..."    # print everything after the final "/"
    process_sequence $seq_dir
done

echo -e "\n\nSuccess: ${success}\nFailed: ${failed}"
echo -e "\n\nSuccess: ${success}\nFailed: ${failed}" >> $status_file