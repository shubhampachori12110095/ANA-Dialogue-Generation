import sys
import nltk

reference_file = open(sys.argv[1],"r")
hypothesis_file = open(sys.argv[2],"r")

references = reference_file.readlines()
references = [reference.strip() for reference in references]
hypothesis = hypothesis_file.readlines()
hypothesis = [hypo.strip() for hypo in hypothesis]

#We are using sentence level smoothing for the bleu score. We apply the method7() introduced in this paper http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
SF = nltk.translate.bleu_score.SmoothingFunction()
score = 0.0
index = 0
for each in references:
	reference = each
	hypo = hypothesis[index]
	reference_list = reference.split()
	hypo_list = hypo.split()
	score += nltk.translate.bleu_score.sentence_bleu([reference_list], hypo_list, smoothing_function = SF.method7)
	index += 1


average_score = float(score) / float(index)

print "Average BLEU score is: " + str(average_score)
