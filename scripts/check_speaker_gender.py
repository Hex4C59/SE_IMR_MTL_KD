def count_train_speakers():
    # 读取Speaker_ids.txt文件，构建说话人ID到性别的映射
    speaker_to_gender = {}
    audio_to_speaker = {}

    # 读取Speaker_ids文件
    with open("scripts/Speaker_ids.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if ";" in line:
                parts = line.split(";")
                if len(parts) == 2:
                    speaker_part = parts[0].strip()
                    info_part = parts[1].strip()

                    # 处理说话人性别信息（格式：Speaker_X; Gender）
                    if speaker_part.startswith("Speaker_") and info_part in [
                        "Male",
                        "Female",
                    ]:
                        speaker_to_gender[speaker_part] = info_part

                    # 处理音频文件到说话人ID的映射（格式：audio_file.wav;speaker_id）
                    elif speaker_part.endswith(".wav") and info_part.isdigit():
                        audio_to_speaker[speaker_part] = f"Speaker_{info_part}"

    # 读取Partitions.txt文件，获取训练集音频文件
    train_audio_files = set()

    with open("scripts/Partitions.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Train;"):
                # 提取音频文件名
                audio_file = line.replace("Train;", "").strip()
                train_audio_files.add(audio_file)

    # 统计训练集中的说话人
    train_speakers = set()
    for audio_file in train_audio_files:
        if audio_file in audio_to_speaker:
            speaker_id = audio_to_speaker[audio_file]
            train_speakers.add(speaker_id)

    # 统计男性和女性说话人数量
    male_speakers = set()
    female_speakers = set()

    for speaker in train_speakers:
        if speaker in speaker_to_gender:
            if speaker_to_gender[speaker] == "Male":
                male_speakers.add(speaker)
            elif speaker_to_gender[speaker] == "Female":
                female_speakers.add(speaker)

    # 输出结果
    print("训练集统计结果:")
    print(f"总训练样本数: {len(train_audio_files)}")
    print(f"涉及说话人总数: {len(train_speakers)}")
    print(f"男性说话人数量: {len(male_speakers)}")
    print(f"女性说话人数量: {len(female_speakers)}")
    print(
        f"未知性别说话人数量: {len(train_speakers) - len(male_speakers) - len(female_speakers)}"
    )

    # 详细信息（可选）
    print("\n详细信息:")
    print(f"男性说话人ID: {sorted(male_speakers)}")
    print(f"女性说话人ID: {sorted(female_speakers)}")

    # 找出未知性别的说话人
    unknown_speakers = train_speakers - male_speakers - female_speakers
    if unknown_speakers:
        print(f"未知性别说话人ID: {sorted(unknown_speakers)}")

    return {
        "total_samples": len(train_audio_files),
        "total_speakers": len(train_speakers),
        "male_speakers": len(male_speakers),
        "female_speakers": len(female_speakers),
        "unknown_speakers": len(unknown_speakers),
    }


if __name__ == "__main__":
    result = count_train_speakers()
