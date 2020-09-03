#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from wildfires.utils import handle_array_job_args

try:
    # This will only work after the path modification carried out in the job script.
    from specific import (
        CACHE_DIR,
        SimpleCache,
        get_model,
        data_split_cache,
        get_shap_values,
    )
except ImportError:
    """Not running as an HPC job yet."""


def func():
    # Used to re-compute specific failed jobs, `None` otherwise.
    indices = [
        1017,
        1088,
        1089,
        1136,
        1174,
        1202,
        1277,
        1303,
        1364,
        1480,
        1529,
        1589,
        1687,
        1805,
        1845,
        1889,
        1896,
        1931,
        1932,
        1941,
        1966,
        1967,
        1970,
        1984,
        2014,
        2035,
        2043,
        2047,
        2051,
        2061,
        2062,
        2067,
        2069,
        2075,
        2081,
        2094,
        2095,
        2103,
        2122,
        2123,
        2126,
        2127,
        2128,
        2130,
        2131,
        2137,
        2138,
        2152,
        2154,
        2155,
        2156,
        2157,
        2159,
        2160,
        2161,
        2173,
        2174,
        2175,
        2176,
        2179,
        2180,
        2188,
        2189,
        2191,
        2193,
        2194,
        2195,
        2196,
        2198,
        2199,
        2202,
        2203,
        2205,
        2206,
        2207,
        2210,
        2224,
        2225,
        2227,
        2232,
        2233,
        2236,
        2238,
        2248,
        2249,
        2250,
        2251,
        2252,
        2253,
        2254,
        2258,
        2259,
        2261,
        2262,
        2263,
        2264,
        2272,
        2273,
        2274,
        2276,
        2279,
        2280,
        2281,
        2282,
        2283,
        2285,
        2287,
        2288,
        2289,
        2291,
        2292,
        2294,
        2302,
        2308,
        2310,
        2314,
        2315,
        2316,
        2320,
        2321,
        2323,
        2327,
        2328,
        2332,
        2340,
        2341,
        2355,
        2356,
        2361,
        2366,
        2367,
        2368,
        2369,
        2371,
        2372,
        2373,
        2377,
        2381,
        2394,
        2395,
        2397,
        2402,
        2403,
        2407,
        2412,
        2418,
        2425,
        2438,
        2441,
        2442,
        2444,
        2456,
        2473,
        2475,
        2476,
        2477,
        2478,
        2494,
        2507,
        2515,
        2516,
        2517,
        2531,
        2534,
        2565,
        2567,
        2571,
        2572,
        2573,
        2574,
        2575,
        2582,
        2587,
        2588,
        2594,
        2595,
        2611,
        2624,
        2625,
        2626,
        2627,
        2628,
        2629,
        2630,
        2632,
        2633,
        2634,
        2637,
        2638,
        2639,
        2640,
        2641,
        2642,
        2650,
        2651,
        2654,
        2655,
        2671,
        2672,
        2673,
        2674,
        2675,
        2676,
        2677,
        2679,
        2680,
        2682,
        2683,
        2686,
        2687,
        2688,
        2689,
        2690,
        2692,
        2697,
        2707,
        2708,
        2710,
        2711,
        2712,
        2714,
        2715,
        2716,
        2717,
        2718,
        2719,
        2721,
        2722,
        2723,
        2724,
        2725,
        2727,
        2730,
        2736,
        2737,
        2757,
        2758,
        2759,
        2762,
        2763,
        2764,
        2765,
        2766,
        2767,
        2768,
        2769,
        2771,
        2772,
        2773,
        2775,
        2776,
        2777,
        2779,
        2780,
        2781,
        2782,
        2783,
        2784,
        2800,
        2801,
        2802,
        2803,
        2804,
        2805,
        2806,
        2807,
        2808,
        2809,
        2810,
        2811,
        2812,
        2813,
        2815,
        2816,
        2817,
        2818,
        2819,
        2820,
        2821,
        2823,
        2824,
        2825,
        2826,
        2831,
        2832,
        2833,
        2834,
        2835,
        2852,
        2854,
        2855,
        2856,
        2857,
        2858,
        2859,
        2861,
        2862,
        2863,
        2864,
        2865,
        2866,
        2867,
        2868,
        2870,
        2871,
        2872,
        2873,
        2874,
        2876,
        2879,
        2880,
        2881,
        2885,
        2900,
        2910,
        2911,
        2912,
        2913,
        2914,
        2915,
        2918,
        2919,
        2920,
        2921,
        2924,
        2925,
        2930,
        2931,
        2932,
        2933,
        2935,
        2936,
        2937,
        2938,
        2943,
        2945,
        2957,
        2961,
        2962,
        2964,
        2965,
        2966,
        2970,
        2971,
        2972,
        2975,
        2976,
        2978,
        2979,
        2980,
        2981,
        2982,
        2985,
        2986,
        2997,
        3005,
        3014,
        3017,
        3023,
        3024,
        3027,
        3028,
        3043,
        3054,
        3061,
        3062,
        3103,
        3141,
        3873,
        3881,
        3884,
        3886,
        3887,
        3890,
        3891,
        3894,
        3895,
        3896,
        3897,
        3898,
        3899,
        3900,
        3903,
        3909,
        3910,
        3911,
        3912,
        3916,
        3925,
        3926,
        3927,
        3928,
        3929,
        3932,
        3933,
        3934,
        3935,
        3936,
        3938,
        3941,
        3942,
        3943,
        3944,
        3947,
        3948,
        3951,
        3952,
        3953,
        3954,
        3955,
        3956,
        3957,
        3958,
        3959,
        3960,
        3961,
        3962,
        3963,
        3964,
        3965,
        3966,
        3967,
        3968,
        3969,
        3970,
        3971,
        3973,
        3974,
        3975,
        3976,
        3978,
        3979,
        3980,
        3981,
        3982,
        3983,
        3984,
        3985,
        3986,
        3987,
        3988,
        3989,
        3990,
        3991,
        3992,
        3993,
        3994,
        3995,
        3996,
        3997,
        3998,
        3999,
        4000,
        4001,
        4002,
        4003,
        4004,
        4005,
        4006,
        4007,
        4015,
        4016,
        4250,
        4284,
        4285,
        4373,
        4374,
        4418,
        4419,
        4420,
        4421,
        4445,
        4446,
        4493,
        4494,
        4495,
        4496,
        4514,
        4515,
        4537,
        4544,
        4545,
        4546,
        4547,
        4570,
        4571,
        4594,
        4600,
        4601,
        4602,
        4603,
        4613,
        4614,
        4625,
        4626,
        4627,
        4631,
        4632,
        4656,
        4657,
        4675,
        4676,
        4693,
        4694,
        4708,
        4709,
        4726,
        4727,
        4743,
        4744,
        4749,
        4750,
        4771,
        4772,
        4788,
        4789,
        4791,
        4792,
        4793,
        4802,
        4803,
        4804,
        4805,
        4815,
        4816,
        4828,
        4829,
        4830,
        4831,
        4832,
        4862,
        4863,
        4873,
        4874,
        4876,
        4877,
        4878,
        4887,
        4888,
        4889,
        4890,
        4897,
        4904,
        4905,
        4925,
        4927,
        4928,
        4929,
        4935,
        4936,
        4937,
        4941,
        4949,
        4950,
        4961,
        4977,
        4985,
        4986,
        4999,
        5007,
        5008,
        5009,
        5025,
        5026,
        5028,
        5029,
        5048,
        5049,
        5057,
        5058,
        5066,
        5067,
        5078,
        5082,
        5083,
        5084,
        5085,
        5104,
        5105,
        5112,
        5113,
        5128,
        5129,
        5133,
        5134,
        5135,
        5136,
        5137,
        5145,
        5146,
        5153,
        5154,
        5168,
        5172,
        5173,
        5174,
        5175,
        5176,
        5178,
        5187,
        5199,
        5200,
        5214,
        5217,
        5221,
        5222,
        5223,
        5224,
        5225,
        5226,
        5227,
        5228,
        5233,
        5242,
        5243,
        5261,
        5266,
        5268,
        5269,
        5270,
        5271,
        5273,
        5279,
        5286,
        5287,
        5293,
        5294,
        5295,
        5296,
        5297,
        5298,
        5299,
        5300,
        5301,
        5302,
        5311,
        5312,
        5320,
        5321,
        5328,
        5329,
        5331,
        5332,
        5333,
        5334,
        5335,
        5336,
        5337,
        5338,
        5343,
        5344,
        5351,
        5352,
        5363,
        5364,
        5368,
        5369,
        5370,
        5371,
        5372,
        5373,
        5374,
        5375,
        5383,
        5384,
        5393,
        5394,
        5402,
        5405,
        5406,
        5407,
        5408,
        5409,
        5410,
        5411,
        5412,
        5413,
        5419,
        5420,
        5430,
        5431,
        5442,
        5446,
        5447,
        5448,
        5449,
        5450,
        5451,
        5452,
        5453,
        5454,
        5456,
        5457,
        5464,
        5465,
        5476,
        5485,
        5486,
        5487,
        5488,
        5489,
        5490,
        5491,
        5492,
        5493,
        5494,
        5499,
        5507,
        5508,
        5524,
        5525,
        5526,
        5527,
        5528,
        5529,
        5530,
        5531,
        5532,
        5542,
        5543,
        5550,
        5551,
        5565,
        5566,
        5572,
        5573,
        5574,
        5575,
        5576,
        5577,
        5578,
        5579,
        5580,
        5581,
        5591,
        5594,
        5595,
        5608,
        5609,
        5612,
        5613,
        5614,
        5620,
        5621,
        5622,
        5623,
        5626,
        5627,
        5628,
        5629,
        5630,
        5640,
        5641,
        5644,
        5645,
        5656,
        5657,
        5658,
        5659,
        5660,
        5662,
        5663,
        5667,
        5668,
        5669,
        5670,
        5671,
        5672,
        5673,
        5674,
        5675,
        5676,
        5681,
        5685,
        5686,
        5713,
        5714,
        5715,
        5716,
        5719,
        5720,
        5729,
        5730,
        5731,
        5732,
        5733,
        5734,
        5735,
        5736,
        5737,
        5738,
        5749,
        5750,
        5756,
        5757,
        5785,
        5786,
        5787,
        5788,
        5791,
        5792,
        5796,
        5799,
        5800,
        5801,
        5802,
        5803,
        5804,
        5805,
        5806,
        5807,
        5823,
        5824,
        5831,
        5832,
        5864,
        5868,
        5869,
        5883,
        5884,
        5901,
        5902,
        5903,
        5904,
        5907,
        5908,
        5911,
        5912,
        5913,
        5914,
        5915,
        5916,
        5917,
        5918,
        5919,
        5920,
        5923,
        5924,
        5932,
        5933,
        5942,
        5943,
        5944,
        5945,
        5946,
        5948,
        5949,
        5950,
        5977,
        5978,
        5979,
        5980,
        5981,
        5983,
        5984,
        5991,
        5995,
        5996,
        5997,
        5999,
    ]

    index = int(os.environ["PBS_ARRAY_INDEX"])

    if indices is not None:
        index = indices[index]

    print("Index:", index)

    X_train, X_test, y_train, y_test = data_split_cache.load()
    rf = get_model()

    job_samples = 50

    tree_path_dependent_shap_interact_cache = SimpleCache(
        f"tree_path_dependent_shap_interact_{index}_{job_samples}",
        cache_dir=os.path.join(CACHE_DIR, "shap_interaction"),
    )

    @tree_path_dependent_shap_interact_cache
    def cached_get_interact_shap_values(model, X):
        return get_shap_values(model, X, interaction=True)

    cached_get_interact_shap_values(
        rf, X_train[index * job_samples : (index + 1) * job_samples]
    )


if __name__ == "__main__":
    handle_array_job_args(
        Path(__file__).resolve(),
        func,
        ncpus=1,
        mem="7gb",
        walltime="11:00:00",
        max_index=859,
    )
