from datetime import datetime

from mrjob.job import MRJob

from dictionary import positive, negative


class MRBaseline(MRJob):

    def mapper(self, _, line):
        line = line.replace('null', '')
        line.strip()
        line = line.replace('\"', '')
        tokens = line.split()
        tokens = [token.strip() for token in tokens]
        pos_count, neg_count = 0, 0
        for word in tokens:
            if word in positive:
                pos_count += 1
            elif word in negative:
                neg_count += 1

        prediction = "prediction" if pos_count >= neg_count else "negative"

        yield prediction, None


if __name__ == '__main__':
    start_time = datetime.now()
    MRBaseline.run()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Total Executing Time : ", elapsed_time)
