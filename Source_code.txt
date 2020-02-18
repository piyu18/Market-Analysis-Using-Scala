
import org.apache.spark.ml.feature.StringIndexer

val bank_people_data = spark.read.option("multiline","true").json("/user/ravichaubey43_gmail/bank_edited.json");
bank_people_data.show()

bank_people_data.registerTempTable("datanewtable")

bank_people_data.select(max($"age")).show()
bank_people_data.select(min($"age")).show()
bank_people_data.select(avg($"age")).show() 
bank_people_data.select(avg($"balance")).show()
val median = spark.sql("SELECT percentile_approx(balance, 0.5) FROM datanewtable").show() 

val agedata = spark.sql("select age, count(*) as number from datanewtable where y='yes' group by age order by number desc")
agedata.show()

val maritaldata = spark.sql("select marital, count(*) as number from datanewtable where y='yes' group by marital order by number desc")
maritaldata.show()

val ageandmaritaldata = spark.sql("select age, marital, count(*) as number from datanewtable where y='yes' group by age,marital order by number desc")
ageandmaritaldata.show()

val agedata = spark.udf.register("agedata",(age:Int) => {
if (age < 20)
"Teen"
else if (age > 20 && age <= 32)
"Young"
else if (age > 33 && age <= 55)
"Middle Aged"
else
"old"
})

//Replacing the old age column with the new age column

val banknewDF = bank_people_data.withColumn("age",agedata(bank_people_data("age")))
banknewDF.show()

banknewDF.registerTempTable("banknewtable")

//which age group subscribed the most

val targetage = spark.sql("select age, count(*) as number from banknewtable where y='yes' group by age order by number desc")
targetage.show()

//pipelining with string Indexer

val agedata2 = new StringIndexer().setInputCol("age").setOutputCol("ageindex")

//Fitting the model

var strindModel = agedata2.fit(banknewDF)

//assigns generated value of index of the column, by feature engineering

strindModel.transform(banknewDF).select("age","ageIndex").show(5)
 
