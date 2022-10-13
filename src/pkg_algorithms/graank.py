import so4gp as sgp


def execute(f_path, min_supp, cores, eq=False):
    try:
        out_json, d_set = sgp.graank(f_path, min_supp, eq, return_obj=True)
        lst_gp = d_set.gradual_patterns

        if cores > 1:
            num_cores = cores
        else:
            num_cores = sgp.get_num_cores()

        wr_line = "Algorithm: GRAANK \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.row_count) + '\n'
        wr_line += "Minimum support: " + str(min_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(lst_gp)) + '\n\n'

        for txt in d_set.titles:
            wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in lst_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        return wr_line
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line
