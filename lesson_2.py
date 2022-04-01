import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    penguins = sns.load_dataset("penguins")
    df = sns.load_dataset("penguins")
    sns.pairplot(df, hue="species")
    plt.show()

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
    chart.save('../Results/lesson2_1.html')

    alt.layer(chart.interactive(), *polynomial_fit).save('../Results/lesson2_2.html')


if __name__=='__main__':
    main()
