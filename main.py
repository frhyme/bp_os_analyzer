import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import random as rnd
import datetime as dt
from markdown import markdown
import pdfkit

def return_sample_log_as_df():
    resource = pd.DataFrame({"id": ["r0","r01", "r02", "r011", "r012", "r021", "r022", "r0211", "r0212" ]})
    resource["h_position"] = [ len(r)-1 for r in resource["id"] ]
    resource["d_id"]= [ "strategy" if "r01" in r else "production" if "r02" in r else "management" for r in resource["id"] ]
    resource["manager_id"] = [r[:-1] if r!="r0" else None for r in resource["id"]]

    OS = nx.Graph()
    OS.add_edges_from([(resource.iloc()[i]["id"], resource.iloc()[i]["manager_id"]) for i in range(len(resource))])
    #print(OS.edges())
    #print(nx.shortest_path(OS, "r011", "r0212"))
    #print( len(nx.shortest_path(OS, "r02", "r02")) )

    max_case_num = 500
    r_num = 3
    cols=["case_id", "resource_id", "role"]
    process_resource = pd.DataFrame(columns=cols)
    #print(process_resource)
    for case_id in range(0, max_case_num):
        r_col = sorted(rnd.sample(list(resource["id"]), r_num), key = lambda x: len(x))
        role_col = ["assigner", "assignee", "approver"]
        case_col = ["case"+str(case_id)]*3
        #$print( pd.DataFrame({"case_id":case_col, "resource_id":r_col, "role":role_col}))
        process_resource = process_resource.append(pd.DataFrame({"case_id":case_col, "resource_id":r_col, "role":role_col}))
    #print(process_resource)

    task_execution_log = pd.DataFrame()
    for i in range(0, len(process_resource), 3):
        time_lst = [dt.datetime(2017, rnd.randint(3, 6), rnd.randint(1, 30),rnd.randint(0,23), rnd.randint(0, 59), 0) for i in range(0, 6)]
        time_lst = sorted(time_lst)
        #print(time_lst)
        for j in range(0, 3):
            added_row = process_resource.iloc()[i+j].to_dict()
            added_row = dict(added_row)
            added_row.update({"start_time": time_lst[j*2], "end_time": time_lst[j*2+1]})
            added_row = {key:[ added_row[key] ] for key in added_row.keys()}
            task_execution_log = task_execution_log.append(pd.DataFrame(pd.DataFrame.from_dict(added_row)))
    #task_execution_log = task_execution_log.reset_index()
    log_for_analysis = pd.DataFrame()
    # task execution
    for i in range(0, len(task_execution_log)):
        row_df = pd.DataFrame()
        row_df["case_id"] = [ task_execution_log.iloc()[i]["case_id"] ]
        row_df["from_resource_id"] = [ task_execution_log.iloc()[i]["resource_id"] ]
        row_df["to_resource_id"] = [ task_execution_log.iloc()[i]["resource_id"] ]
        row_df["from_role"] = [ task_execution_log.iloc()[i]["role"] ]
        row_df["to_role"] = [ task_execution_log.iloc()[i]["role"] ]
        row_df["start_time"] = [ task_execution_log.iloc()[i]["start_time"] ]
        row_df["end_time"] = [ task_execution_log.iloc()[i]["end_time"] ]
        row_df["task_type"] = [ "task_execution"]

        log_for_analysis = log_for_analysis.append(pd.DataFrame.from_dict(row_df))
        """
        added_row = process_resource.iloc()[i+j].to_dict()
            added_row.update({"start_time": time_lst[j*2], "end_time": time_lst[j*2+1]})
            added_row = {key:[ added_row[key] ] for key in added_row.keys()}
            task_execution_log = task_execution_log.append(pd.DataFrame(pd.DataFrame.from_dict(added_row)))
        """
    for i in range(0, len(log_for_analysis)-1):
        if log_for_analysis.iloc()[i]["case_id"]==log_for_analysis.iloc()[i+1]["case_id"]:
            row_df = pd.DataFrame()
            row_df["case_id"] = [ task_execution_log.iloc()[i]["case_id"] ]
            row_df["from_resource_id"] = [ task_execution_log.iloc()[i]["resource_id"] ]
            row_df["to_resource_id"] = [ task_execution_log.iloc()[i+1]["resource_id"] ]
            row_df["from_role"] = [ task_execution_log.iloc()[i]["role"] ]
            row_df["to_role"] = [ task_execution_log.iloc()[i+1]["role"] ]
            row_df["start_time"] = [ task_execution_log.iloc()[i]["end_time"] ]
            row_df["end_time"] = [ task_execution_log.iloc()[i+1]["start_time"] ]
            row_df["task_type"] = [ "transfer_of_task"]

            log_for_analysis = log_for_analysis.append(pd.DataFrame.from_dict(row_df))
        else:
            continue
    log_for_analysis = log_for_analysis.sort_values(["case_id", "start_time"], ascending=[True, True])

    return log_for_analysis

def plot_dotted_chart(df, marker_clmn):
    df = df.sort_values(["case_id", "end_time"], ascending=[1, 1])
    scatter_df=pd.DataFrame({"case_id":df["case_id"], "end_time": df["end_time"], "marker":df[marker_clmn]})
    scatter_df=scatter_df

    sorted_case_id = list( scatter_df[["case_id", "end_time"]].groupby("case_id").min().sort_values("end_time", ascending=False).index )
    caseid_to_i_dict = {sorted_case_id[i]: i for i in range(0, len(sorted_case_id))}
    scatter_df["case_id"] = scatter_df["case_id"].apply(lambda case_id: caseid_to_i_dict[case_id])

    sorted_marker = [x[0] for x in Counter(list(scatter_df["marker"])).most_common()]

    num_clr = int(round(len(sorted_marker)**(1/3) +0.5, 0)) - 1

    clr_lst = [ ( i*((1.0)/num_clr), j*((1.0)/num_clr), k*((1.0)/num_clr), 1.0 ) for i in range(0, num_clr+1) for j in range(0, num_clr+1) for k in range(0, num_clr+1) ]
    from random import shuffle
    shuffle(clr_lst)

    marker_to_clr_dict = {sorted_marker[i]: clr_lst[i] for i in range(0, len(sorted_marker))}
    scatter_df["marker"] = scatter_df["marker"].apply(lambda marker: marker_to_clr_dict[marker])

    #print( scatter_df[:10])
    #cmap을 활용
    dotted_chart = plt.figure(marker_clmn+"_dotted_chart")
    plt.scatter( list( scatter_df["end_time"]), list( scatter_df["case_id"]), c=list( scatter_df["marker"]), s=10, linewidth=0.0)
    plt.ylim( 0.0, len( sorted_case_id ) )
    plt.savefig( "resource_dotted_chart.png")
    #plt.show()

def return_resource_matrix_as_df(df, perspective):
    uniq_r_id_lst =list(set( list(df["to_resource_id"]) + list(df["from_resource_id"]) ))
    uniq_r_id_lst = sorted(uniq_r_id_lst)
    uniq_r_id_lst = sorted(uniq_r_id_lst, key=lambda x: len(x))
    val_dict = {r_id:{r_id:0 for r_id in uniq_r_id_lst} for r_id in uniq_r_id_lst}
    if perspective == "count":
        for i in range(0, len(df)):
            val_dict[df.iloc()[i]["from_resource_id"]][df.iloc()[i]["to_resource_id"]]+=1
        return pd.DataFrame(val_dict)
    elif perspective == "sum_time":
        for i in range(0, len(df)):
            days = (df.iloc()[i]["end_time"] - df.iloc()[i]["start_time"]).total_seconds()/(24*3600)
            val_dict[df.iloc()[i]["from_resource_id"]][df.iloc()[i]["to_resource_id"]]+=days
        val_dict={key1:{ key2:int(val_dict[key1][key2]) for key2 in val_dict[key1].keys()}for key1 in val_dict.keys()}
        return pd.DataFrame(val_dict)
    elif perspective == "average":
        val_dict_count = {r_id:{r_id:0 for r_id in uniq_r_id_lst} for r_id in uniq_r_id_lst}
        val_dict_sum_time = {r_id:{r_id:0 for r_id in uniq_r_id_lst} for r_id in uniq_r_id_lst}
        for i in range(0, len(df)):
            days = (df.iloc()[i]["end_time"] - df.iloc()[i]["start_time"]).total_seconds()/(24*3600)
            val_dict_sum_time[df.iloc()[i]["from_resource_id"]][df.iloc()[i]["to_resource_id"]]+=days
            val_dict_count[df.iloc()[i]["from_resource_id"]][df.iloc()[i]["to_resource_id"]]+=1
        return_df = ( pd.DataFrame(val_dict_sum_time)/pd.DataFrame(val_dict_count) ).fillna(0)
        return return_df.apply(lambda series: ([int(x) for x in series]))
    else:
        print("use wrong persepctive")
        return False
        # matrix elemental operation
def show_and_save_heatmap(heatmap_df, filename):
    cell_dict = heatmap_df.to_dict()
    max_value = max( [ cell_dict[key1][key2] for key1 in cell_dict.keys() for key2 in cell_dict[key1].keys() ] )
    min_value = min( [ cell_dict[key1][key2] for key1 in cell_dict.keys() for key2 in cell_dict[key1].keys() ] )
    f, ax = plt.subplots(figsize=(11,9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(heatmap_df, cmap="Reds", vmax=max_value, vmin=min_value , linewidths=2.5 ,annot=True, ax=ax, fmt="d")
    ax.set_xticklabels(heatmap_df.columns , rotation=90)
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)#
    plt.xticks(rotation=0)#
    #ax.set_xticks(np.arange(0, cell_size))
    #ax.set_yticks(np.arange(0, cell_size))
    plt.tight_layout()
    plt.savefig(filename+"_heatmap.png")

def make_report(file_path, log_for_analysis):
    path_wkthmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'

    # wrtie figure
    input_filename = 'analysis_report.md'
    output_filename = 'analysis_report.pdf'
    with open(input_filename, "w") as f:
        ### write pofi
        # H-pofi(count)
        # print( sum(sum(pd.DataFrame({"a":[1,2], "b":[3, 4]}).values)) )
        h_pofi_count = sum(sum((return_resource_matrix_as_df(log_for_analysis[log_for_analysis["handover matrix"]==True], "count").values)))
        oc_pofi_count = sum(sum((return_resource_matrix_as_df(log_for_analysis[log_for_analysis["org chart matrix"]==True], "count").values)))
        h_pofi_sum_time = sum(sum((return_resource_matrix_as_df(log_for_analysis[log_for_analysis["handover matrix"]==True], "sum_time").values)))
        oc_pofi_sum_time = sum(sum((return_resource_matrix_as_df(log_for_analysis[log_for_analysis["org chart matrix"]==True], "sum_time").values)))
        h_pofi_average = sum(sum((return_resource_matrix_as_df(log_for_analysis[log_for_analysis["handover matrix"]==True], "average").values)))
        oc_pofi_average = sum(sum((return_resource_matrix_as_df(log_for_analysis[log_for_analysis["org chart matrix"]==True], "average").values)))

        overall_fit_index_count = oc_pofi_count/h_pofi_count
        overall_fit_index_sum_time = oc_pofi_sum_time/h_pofi_sum_time
        overall_fit_index_average = oc_pofi_average/h_pofi_average
        f.write("# H-POFI(count): "+str(h_pofi_count))
        f.write("\n")
        f.write("# OC-POFI(count): "+str(oc_pofi_count))
        f.write("\n")
        f.write("# Over fit index(count): "+str(overall_fit_index_count))
        f.write("\n")
        f.write("# H-POFI(sum of time): "+str(h_pofi_sum_time))
        f.write("\n")
        f.write("# OC-POFI(sum of time): "+str(oc_pofi_sum_time))
        f.write("\n")
        f.write("# Over fit index(sum of time): "+str(overall_fit_index_sum_time))
        f.write("\n")
        f.write("# H-POFI(average): "+str(h_pofi_average))
        f.write("\n")
        f.write("# OC-POFI(average): "+str(oc_pofi_average))
        f.write("\n")
        f.write("# Over fit index(average): "+str(overall_fit_index_average))
        f.write("\n")
        f.write("</br>")
        f.write("\n")
        ### overall
        ### write task capacity
        temp = return_resource_matrix_as_df(log_for_analysis, "average")
        temp = pd.DataFrame([temp.iloc()[i][i] for i in range(0, len(temp))], index=temp.columns, columns=["task capacity of each resource"])
        temp = temp.sort_values("task capacity of each resource", ascending=False)
        show_and_save_heatmap(temp, "task_capacity")
        f.write("# Task capacity of each resource")
        f.write("\n")
        f.write("![Task capacity of each resource]("+file_path+"task_capacity_heatmap.png)")
        f.write("\n")

        ### write dotted chart
        plot_dotted_chart(log_for_analysis[log_for_analysis["task_type"]=="task_execution"], "to_resource_id")
        f.write("# Resource dotted chart")
        f.write("\n")
        f.write("#### x axis: case id, y axis: time color: resource")
        f.write("\n")
        f.write("![Resource dotted chart]("+file_path+"resource_dotted_chart.png)")
        f.write("\n")

        ### workflow matrixS
        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis, "count"), "wfm_count")
        print("workflow matrix(count) complete")
        f.write("# Workflow Matrix")
        f.write("\n")
        f.write("## Workflow Matrix(count)")
        f.write("\n")
        f.write("![Wfm_count]("+file_path+"wfm_count_heatmap.png)")
        f.write("\n")

        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis, "sum_time"), "wfm_sum_time")
        print("workflow matrix(sum of time) complete")
        f.write("## Workflow Matrix(sum_time)")
        f.write("\n")
        f.write("![Wfm_sum_time]("+file_path+"wfm_sum_time_heatmap.png)")
        f.write("\n")

        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis, "average"), "wfm_average")
        print("workflow matrix(average) complete")
        f.write("## Workflow Matrix(average)")
        f.write("\n")
        f.write("![Wfm_average]("+file_path+"wfm_average_heatmap.png)")
        f.write("\n")

        ###handover matrix
        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis[log_for_analysis["handover matrix"]==True], "count"), "handover_count")
        print("handover matrix(count) complete")
        f.write("# Handover Matrix")
        f.write("\n")
        f.write("## Handover Matrix(count)")
        f.write("\n")
        f.write("![handover_count]("+file_path+"handover_count_heatmap.png)")
        f.write("\n")

        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis[log_for_analysis["handover matrix"]==True], "sum_time"), "handover_sum_time")
        print("handover matrix(sum of time) complete")
        f.write("## handover Matrix(sum_time)")
        f.write("\n")
        f.write("![handover_sum_time]("+file_path+"handover_sum_time_heatmap.png)")
        f.write("\n")

        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis[log_for_analysis["handover matrix"]==True], "average"), "handover_average")
        print("handover matrix(average) complete")
        f.write("## handover Matrix(average)")
        f.write("\n")
        f.write("![handover_average]("+file_path+"handover_average_heatmap.png)")
        f.write("\n")

        ###org chart matrix
        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis[log_for_analysis["org chart matrix"]==True], "count"), "org_chart_count")
        print("organizational chart matrix(count) complete")
        f.write("# organizational chart Matrix")
        f.write("\n")
        f.write("## organizational chart Matrix(count)")
        f.write("\n")
        f.write("![org_chart_count]("+file_path+"org_chart_count_heatmap.png)")
        f.write("\n")

        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis[log_for_analysis["org chart matrix"]==True], "sum_time"), "org_chart_sum_time")
        print("organizational chart matrix(sum of time) complete")
        f.write("## organizational chart Matrix(sum_time)")
        f.write("\n")
        f.write("![org_chart_sum_time]("+file_path+"org_chart_sum_time_heatmap.png)")
        f.write("\n")

        show_and_save_heatmap(return_resource_matrix_as_df(log_for_analysis[log_for_analysis["org chart matrix"]==True], "average"), "org_chart_average")
        print("organizational chart matrix(average) complete")
        f.write("## organizational chart Matrix(average)")
        f.write("\n")
        f.write("![org_chart_average]("+file_path+"org_chart_average_heatmap.png)")
        f.write("\n")
    f.close()

    with open(input_filename, 'r') as f:
        html_text = markdown(f.read(), output_format='html4')
    config = pdfkit.configuration(wkhtmltopdf=path_wkthmltopdf)
    pdfkit.from_string(html_text, output_filename, configuration=config)

## main
## main
## main
# log_for_analysis = return_sample_log_as_df()

file_path = input("file path:")
resource_csv_file_name = input("resource csv file name:")
log_csv_file_name = input("log csv file name:")


#resource = pd.DataFrame({"id": ["r0","r01", "r02", "r011", "r012", "r021", "r022", "r0211", "r0212" ]})
#resource["h_position"] = [ len(r)-1 for r in resource["id"] ]
#resource["d_id"]= [ "strategy" if "r01" in r else "production" if "r02" in r else "management" for r in resource["id"] ]
#resource["manager_id"] = [r[:-1] if r!="r0" else None for r in resource["id"]]
#resource.to_csv("resource.csv", index=False)

#resource_csv_file_name = "resource.csv"
resource = pd.read_csv(resource_csv_file_name)

OS = nx.Graph()
OS.add_edges_from([(resource["id"].iloc()[i], resource["manager_id"].iloc()[i]) for i in range(0, len(resource))])


#log_csv_file_name = "log.csv"

#print(log_for_analysis[:5])
#log_for_analysis.to_csv(csv_file_name, index=False)
log_for_analysis = pd.read_csv(log_csv_file_name)
print("each name of column should be matched to these")
print(list(log_for_analysis.columns))


log_for_analysis["start_time"] = [ dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in log_for_analysis["start_time"]]
log_for_analysis["end_time"] = [ dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in log_for_analysis["end_time"]]


log_for_analysis["handover matrix"] = [True if log_for_analysis.iloc()[i]["to_resource_id"]!=log_for_analysis.iloc()[i]["from_resource_id"] else False for i in range(0, len(log_for_analysis))]
log_for_analysis["org chart matrix"] = [True if len(nx.shortest_path(OS, log_for_analysis.iloc()[i]["to_resource_id"], log_for_analysis.iloc()[i]["from_resource_id"]))==2 else False for i in range(0, len(log_for_analysis))]

make_report(file_path, log_for_analysis)
