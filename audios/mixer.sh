while read -r clean noise; do
    clean_path=$(echo $clean | tr -d "'")
    noise_path=$(echo $noise | tr -d "'")
    filename=$(basename "$clean_path")

    # [0:a] é o primeiro arquivo (clean), [1:a] é o segundo (noise)
    # Aqui reduzimos o volume do ruído para 0.3 (30%)
    ffmpeg -i "$clean_path" -i "$noise_path" -filter_complex \
        "[0:a]volume=1.0[v];[1:a]volume=0.7[r];[v][r]amix=inputs=2:duration=longest" \
        "clean+noise/${filename%.*}.wav"
done <lista.txt
