# GSM8K behavioral retention results

| Seed | After GSM8K | Sequential final | CAGD final | CAGD - Seq. |
|---:|---:|---:|---:|---:|
| 3407 | 0.294 | 0.065 | 0.265 | +0.199 |
| 3408 | 0.314 | 0.049 | 0.290 | +0.241 |
| 3409 | 0.306 | 0.066 | 0.259 | +0.193 |

Paired improvement: +0.211 +/- 0.015 SEM; two-way 95% bootstrap interval [+0.185, +0.240].

Predeclared qualitative rule: 215 gains and 14 reversals among seed-3407 questions correct when learned.

## Example 1 (test index 1)

**Question:** A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?

**Target:** 3

**Sequential final:** Assuming the total color of the robe is blue and another color, with the total number of bolts in total, find the total number of bolts that are in total.  10

**CAGD final:** The total number of bolts in the robe is 2 + 2/2 = 2 + 1 = 3 bolts. #### 3

## Example 2 (test index 9)

**Question:** Eliza's rate per hour for the first 40 hours she works each week is $10. She also receives an overtime pay of 1.2 times her regular hourly rate. If Eliza worked for 45 hours this week, how much are her earnings for this week?

**Target:** 460

**Sequential final:** Last week: He paid for his first 40 hours, he gets paid 40(.10)=$4.0 He paid for his second 10 hours, he gets paid 10(.12)=1.2

**CAGD final:** The regular hourly rate is $10. The overtime hourly rate is 1.2*$10 = $12. Eliza's regular hourly pay is $10. The first 40 hours of work are paid at $10 regular rate. The remaining 5 hours of work are paid at $12 overtime rate. The earnings for the first 40 hours are 40*$10 = $400. The earnings for the 5 hours of overtime are 5*$12 = $60. The total earnings for the week are $400 + $60 = $460. #### 460

## Example 3 (test index 14)

**Question:** In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?

**Target:** 60

**Sequential final:** In a dance class of 20 students, 20% enrolled in contemporary dance: This is equal to $0.20 \times 20 = 4$ students. 
25% of the remaining enrolled in jazz dance: The total number of remaining students is 100% $-$ 20% = 80%, or equivalently, 0.80

**CAGD final:** The dance class has 20 students. 20% enrolled in contemporary dance, so 20 students * 20% = 4 students. The remaining students are 20 - 4 = 16 students. Out of these 16 students, 25% enrolled in jazz dance. So, 16 students * 25% = 4 students. The rest of the students enrolled in hip-hop dance, which is 16 - 4 = 12 students. Thus, the percentage of the entire class that enrolled in hip-hop dance is 12 students / 20 students * 100% = 60% #### 60
