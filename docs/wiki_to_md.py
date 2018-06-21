import sys

'''
This script will parse the given .wiki file, tweak some basic parts,
and output a .md file
Pass in the source file and output file in the terminal.
Please adjust imgpath string below.
'''
src_file = sys.argv[1]
output_file = sys.argv[2]
imgpath = "../../_static/img/tut_corr"
# read a wiki file
output = []
file = open(src_file)
wiki = file.read().splitlines()
for line in wiki:
    if line.startswith("=") and line.endswith("="):
        # handle headings: replace = with #, and add extra #
        line = line.replace("= ", "## ")
        line = line.replace(" =", " ##")
        line = line.replace("=", "#")
        output.append(line)

    elif "'''" in line:
        # handle boldface: replace ''' with **
        line = line.replace("'''", "**")
        line = line.replace("'''", "**")
        output.append(line)

    elif "<i>" in line:
        # handle itatlics: replace <i> and </i> with *
        line = line.replace("<i>", "*")
        line = line.replace("</i>", "*")
        output.append(line)

    elif "[[File" in line:
        # handle images
        start_filename = line.find(':') + 1 
        end_filename = line.find('|', 6)
        output.append('![](%s/%s)'%(imgpath, line[start_filename:end_filename]))

    elif "[http" in line:
        # handle external links
        start_link = line.find('[h') + 1
        end_link = line.find(' ', start_link)
        end_desc = line.find(']', end_link)
        desc = line[end_link+1:end_desc]
        link = line[start_link:end_link]
        newline = line.replace(line[start_link-1:end_desc+1], '[%s](%s)'%(desc, link))
        output.append(newline)

    else:
        output.append(line)

file.close()

# write output to .md file
outfile = open(output_file, 'w+')
for line in output:
    outfile.write('%s\n'%line)
outfile.close()