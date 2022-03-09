import altair as alt
import seaborn as sns

penguins = sns.load_dataset("penguins")
df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
df.head()

chart = alt.Chart(penguins).mark_circle(color="black").encode(
    alt.X('bill_length_mm', scale=alt.Scale(zero=False)),
    alt.Y('bill_depth_mm', scale=alt.Scale(zero=False)),
    color = 'species',
    tooltip = 'bill_depth_mm'
 )

degree_list = [1, 3, 5]

polynomial_fit = [
    chart.transform_regression(
        "bill_length_mm", "bill_depth_mm", method="poly", order=order, as_=["bill_length_mm", str(order)]
    )
    .mark_line()
    .transform_fold([str(order)], as_=["species", "bill_depth_mm"])
    .encode(alt.Color("species:N"))
    for order in degree_list
]

chart.save('/home/michal/Pulpit/Results/res.html')

alt.layer(chart.interactive(), *polynomial_fit)