import sys
import tqdm
class Parser:
    def __init__(self, PARAMS, filepath):
        self.PARAMS = PARAMS
        self.filename = filepath
        self.sequences = dict()
        self.borders = dict()

    def log(self, x):
        print(x, file=sys.stderr)
    
    def parse_file(self, filename):
        """Reads the file, extracts borders and genome sequence."""
        with open(self.filename, "r") as f:
            lines = f.readlines()
        f.close()
        self.log(f"read {len(lines)} lines from file\n")
        # Extract borders
        result = []
        for i in range(0, len(lines), 2):
            border_parts = lines[i].strip().split("[")[1].split("]")[0].split()
            L, R = map(int, border_parts)

            # Extract genome sequence
            self.sequences[i // 2] = lines[i + 1].strip()
            self.borders[i // 2] = (L, R)

            result.append((L, R, i // 2))
        self.log(f"extracted {len(result)} sequences\n")
        return result
    

    def find_intersections(self, segments):
        """Finds intersections between segments and records their indices."""
        intersection_map = dict()

        segs = []

        for i, (start1, end1, index1) in enumerate(segments):
            for j, (start2, end2, index2) in enumerate(segments):
                if i < j:  # Avoid redundant checks
                    intersection_start = max(start1, start2)
                    intersection_end = min(end1, end2)
                    if intersection_start < intersection_end and intersection_end - intersection_start > 15000:
                        segs.append((intersection_start, intersection_end))
        segs =  list(set(segs))
        for (l, r) in segs:
            for (L, R, index) in segments:
                if L <= l and r <= R:
                    if (l, r) not in intersection_map:
                        intersection_map[(l, r)] = []
                    intersection_map[(l, r)].append(index)
        self.log(f"found {len(intersection_map)} intersections\n")
        return intersection_map
    
    def extract_segment_sequence(self, start, end, individual):
        """Extracts the part of the individual's sequence that overlaps with (start, end)."""
        l, r = self.borders[individual]  # Get individual's segment boundaries
        seq = self.sequences[individual]  # Full sequence of individual's segment

        # Compute intersection within individual's segment
        overlap_start = max(l, start)
        overlap_end = min(r, end)

        if overlap_start < overlap_end:  # Valid intersection
            relative_start = overlap_start - l  # Convert to sequence index
            relative_end = overlap_end - l
            return seq[relative_start:relative_end]  # Extract substring

        return ""  # No overlap

    def parse(self):
        segments = self.parse_file(self.filename)
        intersections_map = self.find_intersections(segments)
        intersections = []
        for key in intersections_map.keys():
            intersections.append((key[0], key[1], intersections_map[key]))
        samples = []
        counter = 0
        for start, end, individuals in tqdm.tqdm(intersections):
            fragments = []
            for individual in individuals:
                seq_fragment = self.extract_segment_sequence(start, end, individual)
                '''TODO: change this'''
                if 70_000 >= len(seq_fragment) >= 15_000:
                    fragments.append(seq_fragment)
            res = 1
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    L = len(fragments[i])
                    for start in range(0, L, self.PARAMS['window']):
                        end = start + self.PARAMS["window"]
                        diff = 0
                        for t in range(start, min(end, L)):
                            if fragments[i][t] != fragments[j][t]:
                                diff += 1
                        samples.append((diff, end - start))
            if len(fragments) > 1:
                counter += 1
        self.log(f"found {counter} intersections satisfying criteria")
        return samples