# A demo to use elmo
- [ ] 准备数据
    - [ ] ./data/example文件夹下的txt，是从网上爬取的小说文本
    - [ ] 文本中有很多的奇怪字符，先删掉一批，生成的文件为*_raw.txt, 调用程序为./data/data_preprocessed.py中的`outVocab`接口  
    - [ ] 将*_raw.txt中的大段文本，生成小段文本*_origin.txt
    - [ ] 将*_origin.txt生成分词和分字的文本：*_seg_chars.txt, *_seg_words.txt
    - [ ] 生成字字典和词字典， vocab_raw.txt, vocab_seg_words.txt
- [ ] 训练
    - [ ] 训练：python train_elmo_try.py --save_dir ./log --vocab_file ../data/vocab_seg_words_elmo.txt --train_prefix '../data/example/*_seg_words.txt'
    