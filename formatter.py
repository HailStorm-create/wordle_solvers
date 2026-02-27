with open("wordlesol.txt", "w"):
    pass


with open("words.txt", "r") as infile, open("wordlesol.txt", "a") as outfile:
    for line in infile:
        words = line.strip().split()

        for word in words:
            word = word.upper()

            if len(word) == 5 and word.isalpha():
                outfile.write(word + "\n")
