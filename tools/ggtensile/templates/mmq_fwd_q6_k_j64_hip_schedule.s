// Selected instruction body derived from the project-owned HIP Q6_K control.
// Source: build/mmq_bundle_sources/gfx1151/DenseFwdQ6KK2048J64Full-fb62596f6a0841e0.cu
// Compiler: AMD clang 23.0.0git, revision 998a696feca902c689c53383e48d29494f9edba7.
// Its 47 compiler delay hints are omitted after two bit-exact timing confirmations.
; %bb.0:
	v_bfe_u32 v4, v0, 10, 10
	v_dual_mov_b32 v32, 0 :: v_dual_and_b32 v1, 0x3ff, v0
	v_bfe_u32 v5, v0, 2, 8
	v_dual_mov_b32 v46, 0 :: v_dual_and_b32 v3, 0x70, v0
	v_dual_mov_b32 v44, 0 :: v_dual_lshlrev_b32 v41, 3, v4
	v_dual_mov_b32 v43, 0 :: v_dual_lshlrev_b32 v2, 1, v1
	v_dual_mov_b32 v39, 0 :: v_dual_and_b32 v34, 15, v0
	v_dual_mov_b32 v37, 0 :: v_dual_and_b32 v42, 2, v5
	v_dual_mov_b32 v38, 0 :: v_dual_add_nc_u32 v5, v5, v41
	v_dual_mov_b32 v36, 0 :: v_dual_and_b32 v9, 3, v0
	v_bfe_u32 v0, v0, 4, 6
	s_clause 0x2
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_load_b32 s22, s[0:1], 0x20
	v_lshl_add_u32 v1, v4, 5, v1
	v_sub_nc_u32_e32 v6, v2, v34
	v_dual_mov_b32 v28, 0 :: v_dual_and_b32 v5, 63, v5
	v_lshl_or_b32 v0, v4, 4, v0
	v_dual_mov_b32 v40, 0 :: v_dual_and_b32 v7, 63, v1
	v_lshl_add_u32 v6, v6, 2, 0
	v_mul_u32_u24_e32 v8, 0x130, v4
	v_xor_b32_e32 v11, 32, v5
	v_mul_u32_u24_e32 v12, 0x4c, v0
	v_and_or_b32 v3, v2, 14, v3
	v_dual_mov_b32 v30, 0 :: v_dual_lshlrev_b32 v45, 3, v7
	v_mul_u32_u24_e32 v7, 0x130, v7
	v_dual_mov_b32 v35, 0 :: v_dual_lshlrev_b32 v10, 1, v9
	v_lshl_add_u32 v9, v9, 2, 0
	v_dual_mov_b32 v26, 0 :: v_dual_lshlrev_b32 v47, 3, v5
	v_mul_u32_u24_e32 v5, 0x130, v5
	v_mad_u32_u24 v4, 0x1300, v4, 0
	v_dual_mov_b32 v33, 0 :: v_dual_lshlrev_b32 v48, 3, v11
	v_mul_u32_u24_e32 v11, 0x130, v11
	v_lshl_add_u32 v51, v12, 2, 0
	v_dual_mov_b32 v24, 0 :: v_dual_lshlrev_b32 v53, 1, v2
	v_dual_mov_b32 v29, 0 :: v_dual_add_nc_u32 v2, v6, v8
	v_mad_u32_u24 v49, 0x130, v34, v4
	v_mad_u32_u24 v50, 0x90, v34, 0
	v_lshl_add_u32 v52, v1, 2, 0
	v_dual_mov_b32 v31, 0 :: v_dual_lshlrev_b32 v54, 1, v3
	v_dual_mov_b32 v22, 0 :: v_dual_add_nc_u32 v55, 0, v7
	v_dual_mov_b32 v27, 0 :: v_dual_lshlrev_b32 v56, 1, v10
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v57, v9, v5
	v_dual_mov_b32 v25, 0 :: v_dual_add_nc_u32 v58, v9, v11
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v59, 0x2400, v2
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v60, 0x2800, v2
	v_dual_mov_b32 v16, 0 :: v_dual_add_nc_u32 v61, 0x2c00, v2
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v62, 0x3000, v2
	v_dual_mov_b32 v14, 0 :: v_dual_add_nc_u32 v63, 0x3800, v2
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v64, 0x3c00, v2
	v_dual_mov_b32 v12, 0 :: v_dual_add_nc_u32 v65, 0x4000, v2
	v_dual_mov_b32 v17, 0 :: v_dual_add_nc_u32 v66, 0x4400, v2
	v_add_nc_u32_e32 v67, 0x4800, v2
	v_dual_mov_b32 v15, 0 :: v_dual_add_nc_u32 v68, 0x4e00, v2
	v_add_nc_u32_e32 v69, 0x5400, v2
	v_dual_mov_b32 v13, 0 :: v_dual_add_nc_u32 v70, 0x5800, v2
	v_add_nc_u32_e32 v71, 0x5c00, v2
	v_dual_mov_b32 v11, 0 :: v_dual_add_nc_u32 v72, 0x6000, v2
	v_add_nc_u32_e32 v73, 0x6400, v2
	v_add_nc_u32_e32 v74, 0x6c00, v2
	v_add_nc_u32_e32 v75, 0x2600, v51
	v_add_nc_u32_e32 v76, 0x2a00, v51
	v_add_nc_u32_e32 v77, 0x2e00, v51
	v_add_nc_u32_e32 v78, 0x3400, v51
	s_mov_b32 s4, 0
	s_lshl_b32 s3, s3, 6
	s_lshl_b32 s20, s2, 9
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s21, s22, 1
	s_mul_i32 s22, s22, 36
	s_mov_b32 s5, s4
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	s_mov_b32 s8, s4
	s_mov_b32 s9, s4
	s_mov_b32 s10, s4
	s_mov_b32 s11, s4
	s_mov_b32 s23, s4
.LBB0_1:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_2 Depth 2
                                        ;     Child Loop BB0_4 Depth 2
	s_add_i32 s18, s23, s20
	s_mul_i32 s19, s18, 0xd2
	s_mul_hi_i32 s24, s18, 0xd2
	s_add_u32 s18, s12, s19
	s_addc_u32 s19, s13, s24
	s_mul_i32 s24, s21, s23
	v_mad_u64_u32 v[2:3], null, 0xd2, v41, s[18:19]
	s_add_i32 s24, s24, s3
	v_mad_u64_u32 v[114:115], null, 0xd2, v48, s[18:19]
	v_mad_u64_u32 v[123:124], null, 0xd2, v45, s[18:19]
	v_add_co_u32 v4, vcc_lo, v2, v53
	v_add_co_ci_u32_e64 v5, null, 0, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v2, v54
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	v_add_co_u32 v6, vcc_lo, 0x1000, v4
	v_add_co_ci_u32_e64 v7, null, 0, v5, vcc_lo
	v_add_co_u32 v79, vcc_lo, 0x1000, v2
	v_add_co_ci_u32_e64 v80, null, 0, v3, vcc_lo
	v_add_co_u32 v81, vcc_lo, 0x3000, v4
	v_add_co_ci_u32_e64 v82, null, 0, v5, vcc_lo
	v_add_co_u32 v83, vcc_lo, 0x3000, v2
	v_add_co_ci_u32_e64 v84, null, 0, v3, vcc_lo
	v_add_co_u32 v85, vcc_lo, 0x4000, v4
	v_add_co_ci_u32_e64 v86, null, 0, v5, vcc_lo
	v_add_co_u32 v87, vcc_lo, 0x4000, v2
	v_add_co_ci_u32_e64 v88, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_b32 v10, v[4:5], off
	global_load_b32 v122, v[2:3], off offset:128
	global_load_b32 v8, v[6:7], off offset:2624
	global_load_b32 v79, v[79:80], off offset:2752
	global_load_b32 v7, v[81:82], off offset:1152
	global_load_b32 v9, v[83:84], off offset:1280
	global_load_b32 v6, v[85:86], off offset:3776
	global_load_b32 v80, v[87:88], off offset:3904
	v_add_co_u32 v81, vcc_lo, 0x6000, v4
	v_add_co_ci_u32_e64 v82, null, 0, v5, vcc_lo
	v_add_co_u32 v85, vcc_lo, 0x6000, v2
	v_add_co_ci_u32_e64 v86, null, 0, v3, vcc_lo
	v_add_co_u32 v89, vcc_lo, 0x8000, v4
	v_add_co_ci_u32_e64 v90, null, 0, v5, vcc_lo
	v_add_co_u32 v91, vcc_lo, 0x8000, v2
	v_add_co_ci_u32_e64 v92, null, 0, v3, vcc_lo
	v_add_co_u32 v93, vcc_lo, 0x9000, v4
	v_add_co_ci_u32_e64 v94, null, 0, v5, vcc_lo
	v_add_co_u32 v95, vcc_lo, 0x9000, v2
	v_add_co_ci_u32_e64 v96, null, 0, v3, vcc_lo
	v_add_co_u32 v97, vcc_lo, 0xb000, v4
	v_add_co_ci_u32_e64 v98, null, 0, v5, vcc_lo
	v_add_co_u32 v99, vcc_lo, 0xb000, v2
	v_add_co_ci_u32_e64 v100, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_b32 v84, v[81:82], off offset:2304
	global_load_b32 v88, v[85:86], off offset:2432
	global_load_b32 v83, v[89:90], off offset:832
	global_load_b32 v86, v[91:92], off offset:960
	global_load_b32 v82, v[93:94], off offset:3456
	global_load_b32 v85, v[95:96], off offset:3584
	global_load_b32 v81, v[97:98], off offset:1984
	global_load_b32 v87, v[99:100], off offset:2112
	v_add_co_u32 v89, vcc_lo, 0xd000, v4
	v_add_co_ci_u32_e64 v90, null, 0, v5, vcc_lo
	v_add_co_u32 v91, vcc_lo, 0xd000, v2
	v_add_co_ci_u32_e64 v92, null, 0, v3, vcc_lo
	v_add_co_u32 v93, vcc_lo, 0xe000, v4
	v_add_co_ci_u32_e64 v94, null, 0, v5, vcc_lo
	v_add_co_u32 v95, vcc_lo, 0xe000, v2
	v_add_co_ci_u32_e64 v96, null, 0, v3, vcc_lo
	v_add_co_u32 v97, vcc_lo, 0x10000, v4
	v_add_co_ci_u32_e64 v98, null, 0, v5, vcc_lo
	v_add_co_u32 v99, vcc_lo, 0x10000, v2
	v_add_co_ci_u32_e64 v100, null, 0, v3, vcc_lo
	v_add_co_u32 v101, vcc_lo, 0x12000, v4
	v_add_co_ci_u32_e64 v102, null, 0, v5, vcc_lo
	v_add_co_u32 v103, vcc_lo, 0x12000, v2
	v_add_co_ci_u32_e64 v104, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_b32 v109, v[89:90], off offset:512
	global_load_b32 v113, v[91:92], off offset:640
	global_load_b32 v108, v[93:94], off offset:3136
	global_load_b32 v111, v[95:96], off offset:3264
	global_load_b32 v107, v[97:98], off offset:1664
	global_load_b32 v110, v[99:100], off offset:1792
	global_load_b32 v106, v[101:102], off offset:192
	global_load_b32 v112, v[103:104], off offset:320
	v_add_co_u32 v89, vcc_lo, 0x13000, v4
	v_add_co_ci_u32_e64 v90, null, 0, v5, vcc_lo
	v_add_co_u32 v91, vcc_lo, 0x13000, v2
	v_add_co_ci_u32_e64 v92, null, 0, v3, vcc_lo
	v_add_co_u32 v93, vcc_lo, 0x15000, v4
	v_add_co_ci_u32_e64 v94, null, 0, v5, vcc_lo
	v_add_co_u32 v95, vcc_lo, 0x15000, v2
	v_add_co_ci_u32_e64 v96, null, 0, v3, vcc_lo
	v_add_co_u32 v97, vcc_lo, 0x16000, v4
	v_add_co_ci_u32_e64 v98, null, 0, v5, vcc_lo
	v_add_co_u32 v99, vcc_lo, 0x17000, v2
	v_add_co_ci_u32_e64 v100, null, 0, v3, vcc_lo
	v_add_co_u32 v101, vcc_lo, 0x18000, v2
	v_add_co_ci_u32_e64 v102, null, 0, v3, vcc_lo
	v_mad_u64_u32 v[2:3], null, s24, 36, v[1:2]
	v_mad_u64_u32 v[103:104], null, 0xd2, v47, s[18:19]
	v_add_co_u32 v4, vcc_lo, 0x18000, v4
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_mov_b32 s18, -4
	v_ashrrev_i32_e32 v3, 31, v2
	v_add_co_u32 v103, vcc_lo, v103, v56
	v_add_co_ci_u32_e64 v104, null, 0, v104, vcc_lo
	v_lshlrev_b64 v[116:117], 2, v[2:3]
	v_add_co_u32 v125, vcc_lo, v114, v56
	v_add_co_ci_u32_e64 v126, null, 0, v115, vcc_lo
	v_add_co_u32 v127, vcc_lo, s14, v116
	v_add_co_ci_u32_e64 v128, null, s15, v117, vcc_lo
	s_clause 0xa
	global_load_b32 v120, v[91:92], off offset:2944
	global_load_b32 v116, v[93:94], off offset:1344
	global_load_b32 v119, v[95:96], off offset:1472
	global_load_b32 v117, v[99:100], off
	global_load_b32 v115, v[101:102], off offset:2624
	global_load_b32 v114, v[4:5], off offset:2496
	global_load_b32 v118, v[97:98], off offset:3968
	global_load_b32 v121, v[89:90], off offset:2816
	global_load_d16_b16 v124, v[123:124], off offset:208
	global_load_b32 v3, v[103:104], off offset:192
	global_load_b32 v4, v[125:126], off offset:192
	s_clause 0x7
	global_load_b32 v5, v[127:128], off
	global_load_b32 v89, v[127:128], off offset:512
	global_load_b32 v90, v[127:128], off offset:1024
	global_load_b32 v91, v[127:128], off offset:1536
	global_load_b32 v92, v[127:128], off offset:2048
	global_load_b32 v93, v[127:128], off offset:2560
	global_load_b32 v94, v[127:128], off offset:3072
	global_load_b32 v95, v[127:128], off offset:3584
	v_add_co_u32 v98, vcc_lo, v127, 0x2000
	v_add_co_ci_u32_e64 v99, null, 0, v128, vcc_lo
	v_add_co_u32 v104, vcc_lo, 0x1000, v127
	v_add_co_ci_u32_e64 v105, null, 0, v128, vcc_lo
	v_add_co_u32 v125, vcc_lo, 0x2000, v127
	v_add_co_ci_u32_e64 v126, null, 0, v128, vcc_lo
	s_clause 0x9
	global_load_b32 v97, v[98:99], off offset:-4096
	global_load_b32 v96, v[98:99], off
	global_load_b32 v98, v[104:105], off offset:512
	global_load_b32 v99, v[104:105], off offset:1024
	global_load_b32 v100, v[104:105], off offset:1536
	global_load_b32 v101, v[104:105], off offset:2048
	global_load_b32 v102, v[104:105], off offset:2560
	global_load_b32 v103, v[104:105], off offset:3072
	global_load_b32 v104, v[104:105], off offset:3584
	global_load_b32 v105, v[125:126], off offset:512
	s_waitcnt vmcnt(52)
	v_lshrrev_b32_e32 v123, 4, v10
	s_waitcnt vmcnt(51)
	v_ashrrev_i32_e32 v122, v42, v122
	v_and_b32_e32 v10, 0xf0f0f0f, v10
	s_waitcnt vmcnt(49)
	v_ashrrev_i32_e32 v79, v42, v79
	s_waitcnt vmcnt(48)
	v_and_b32_e32 v128, 0xf0f0f0f, v7
	v_and_b32_e32 v123, 0xf0f0f0f, v123
	s_waitcnt vmcnt(47)
	v_ashrrev_i32_e32 v127, v42, v9
	v_lshrrev_b32_e32 v7, 4, v7
	v_lshrrev_b32_e32 v125, 4, v8
	v_and_b32_e32 v126, 0xf0f0f0f, v8
	s_waitcnt vmcnt(45)
	v_ashrrev_i32_e32 v80, v42, v80
	v_and_b32_e32 v129, 0xf0f0f0f, v6
	v_and_b32_e32 v140, 0xf0f0f0f, v7
	v_lshrrev_b32_e32 v6, 4, v6
	v_lshlrev_b32_e32 v141, 4, v80
	v_and_b32_e32 v143, 0xf0f0f0f, v6
	s_waitcnt vmcnt(44)
	v_lshrrev_b32_e32 v8, 4, v84
	s_waitcnt vmcnt(43)
	v_ashrrev_i32_e32 v88, v42, v88
	s_waitcnt vmcnt(42)
	v_lshrrev_b32_e32 v9, 4, v83
	v_and_b32_e32 v130, 0xf0f0f0f, v84
	s_waitcnt vmcnt(41)
	v_ashrrev_i32_e32 v84, v42, v86
	v_and_b32_e32 v86, 0xf0f0f0f, v83
	s_waitcnt vmcnt(39)
	v_ashrrev_i32_e32 v83, v42, v85
	v_and_b32_e32 v147, 0xf0f0f0f, v9
	v_and_b32_e32 v85, 0xf0f0f0f, v82
	v_lshrrev_b32_e32 v82, 4, v82
	s_waitcnt vmcnt(38)
	v_and_b32_e32 v131, 0xf0f0f0f, v81
	v_lshrrev_b32_e32 v81, 4, v81
	s_waitcnt vmcnt(37)
	v_ashrrev_i32_e32 v87, v42, v87
	v_lshlrev_b32_e32 v144, 4, v88
	v_and_b32_e32 v145, 0xf0f0f0f, v8
	v_and_b32_e32 v82, 0xf0f0f0f, v82
	v_and_b32_e32 v81, 0xf0f0f0f, v81
	v_lshlrev_b32_e32 v146, 4, v84
	v_lshlrev_b32_e32 v148, 4, v83
	v_lshlrev_b32_e32 v149, 4, v87
	v_and_or_b32 v131, 0x30303030, v149, v131
	s_waitcnt vmcnt(36)
	v_and_b32_e32 v132, 0xf0f0f0f, v109
	v_lshrrev_b32_e32 v109, 4, v109
	s_waitcnt vmcnt(34)
	v_and_b32_e32 v133, 0xf0f0f0f, v108
	v_lshrrev_b32_e32 v108, 4, v108
	s_waitcnt vmcnt(32)
	v_and_b32_e32 v134, 0xf0f0f0f, v107
	v_lshrrev_b32_e32 v107, 4, v107
	s_waitcnt vmcnt(30)
	v_and_b32_e32 v135, 0xf0f0f0f, v106
	v_lshrrev_b32_e32 v106, 4, v106
	v_ashrrev_i32_e32 v111, v42, v111
	v_ashrrev_i32_e32 v110, v42, v110
	s_waitcnt vmcnt(29)
	v_ashrrev_i32_e32 v112, v42, v112
	v_ashrrev_i32_e32 v113, v42, v113
	v_and_b32_e32 v109, 0xf0f0f0f, v109
	v_and_b32_e32 v108, 0xf0f0f0f, v108
	v_and_b32_e32 v107, 0xf0f0f0f, v107
	v_and_b32_e32 v106, 0xf0f0f0f, v106
	v_lshlrev_b32_e32 v151, 4, v111
	v_lshlrev_b32_e32 v152, 4, v110
	v_lshlrev_b32_e32 v153, 4, v112
	v_lshlrev_b32_e32 v150, 4, v113
	v_and_or_b32 v149, 0x30303030, v112, v106
	v_and_or_b32 v133, 0x30303030, v151, v133
	v_and_or_b32 v134, 0x30303030, v152, v134
	v_and_or_b32 v132, 0x30303030, v150, v132
	s_waitcnt vmcnt(28)
	v_ashrrev_i32_e32 v120, v42, v120
	s_waitcnt vmcnt(27)
	v_and_b32_e32 v137, 0xf0f0f0f, v116
	s_waitcnt vmcnt(26)
	v_ashrrev_i32_e32 v119, v42, v119
	v_lshrrev_b32_e32 v116, 4, v116
	s_waitcnt vmcnt(25)
	v_ashrrev_i32_e32 v117, v42, v117
	s_waitcnt vmcnt(24)
	v_ashrrev_i32_e32 v115, v42, v115
	s_waitcnt vmcnt(22)
	v_lshrrev_b32_e32 v138, 4, v118
	s_waitcnt vmcnt(21)
	v_and_b32_e32 v136, 0xf0f0f0f, v121
	s_waitcnt vmcnt(20)
	v_cvt_f32_f16_e64 v142, v124.l
	v_lshlrev_b32_e32 v124, 4, v122
	v_and_or_b32 v122, 0x30303030, v122, v123
	v_lshlrev_b32_e32 v123, 4, v79
	v_lshrrev_b32_e32 v121, 4, v121
	v_lshlrev_b32_e32 v155, 4, v119
	v_and_or_b32 v10, 0x30303030, v124, v10
	v_lshlrev_b16 v9.l, 8, v122.h
	v_and_b32_e32 v124, 0xf0f0f0f, v125
	v_lshlrev_b32_e32 v125, 4, v127
	v_lshlrev_b16 v8.l, 8, v122.l
	v_lshlrev_b16 v7.l, 8, v10.h
	v_add_nc_u16 v9.l, 0xe000, v9.l
	v_lshlrev_b16 v6.l, 8, v10.l
	v_and_b16 v7.h, 0x3f00, v10.h
	v_and_b16 v8.h, 0x3f00, v122.l
	v_add_nc_u16 v7.l, 0xe000, v7.l
	v_and_b16 v9.h, 0x3f00, v122.h
	v_and_or_b32 v122, 0x30303030, v123, v126
	v_and_or_b32 v123, 0x30303030, v79, v124
	v_and_or_b32 v124, 0x30303030, v125, v128
	v_and_or_b32 v125, 0x30303030, v127, v140
	v_lshrrev_b16 v7.l, 8, v7.l
	v_lshrrev_b16 v9.l, 8, v9.l
	v_and_or_b32 v126, 0x30303030, v141, v129
	v_and_or_b32 v127, 0x30303030, v80, v143
	v_and_or_b32 v128, 0x30303030, v144, v130
	v_and_or_b32 v129, 0x30303030, v88, v145
	v_and_b32_e32 v121, 0xf0f0f0f, v121
	v_and_b32_e32 v116, 0xf0f0f0f, v116
	v_and_b32_e32 v138, 0xf0f0f0f, v138
	v_and_or_b32 v140, 0x30303030, v84, v147
	v_and_or_b32 v143, 0x30303030, v83, v82
	v_and_or_b32 v144, 0x30303030, v87, v81
	v_add_nc_u16 v6.l, 0xe000, v6.l
	v_add_nc_u16 v8.l, 0xe000, v8.l
	v_lshlrev_b16 v79.l, 8, v122.h
	v_lshlrev_b16 v80.l, 8, v123.l
	v_lshlrev_b16 v81.l, 8, v123.h
	v_lshlrev_b16 v82.l, 8, v124.l
	v_lshlrev_b16 v83.h, 8, v125.l
	v_lshlrev_b16 v84.h, 8, v125.h
	v_or_b16 v7.l, v7.h, v7.l
	v_or_b16 v9.l, v9.h, v9.l
	v_and_b32_e32 v139, 0xf0f0f0f, v114
	v_lshrrev_b32_e32 v114, 4, v114
	v_and_or_b32 v130, 0x30303030, v146, v86
	v_and_or_b32 v141, 0x30303030, v148, v85
	v_and_or_b32 v145, 0x30303030, v113, v109
	v_and_or_b32 v146, 0x30303030, v111, v108
	v_and_or_b32 v147, 0x30303030, v110, v107
	v_lshlrev_b16 v85.l, 8, v126.l
	v_lshlrev_b16 v88.h, 8, v127.h
	v_lshlrev_b16 v106.l, 8, v128.l
	v_lshlrev_b16 v107.h, 8, v128.h
	v_lshlrev_b16 v108.l, 8, v129.l
	v_lshlrev_b16 v109.h, 8, v129.h
	v_and_b32_e32 v118, 0xf0f0f0f, v118
	v_lshlrev_b32_e32 v154, 4, v120
	v_lshlrev_b32_e32 v156, 4, v117
	v_and_b16 v6.h, 0x3f00, v10.l
	v_and_or_b32 v148, 0x30303030, v153, v135
	v_and_or_b32 v151, 0x30303030, v120, v121
	v_and_or_b32 v152, 0x30303030, v155, v137
	v_and_or_b32 v153, 0x30303030, v119, v116
	v_and_or_b32 v155, 0x30303030, v117, v138
	v_and_b16 v81.h, 0x3f00, v123.h
	v_lshlrev_b16 v117.h, 8, v143.h
	v_and_b16 v119.l, 0x3f00, v143.h
	v_lshlrev_b16 v121.h, 8, v144.h
	v_and_b16 v123.h, 0x3f00, v144.h
	v_lshrrev_b16 v6.l, 8, v6.l
	v_lshrrev_b16 v8.l, 8, v8.l
	v_add_nc_u16 v143.h, 0xe000, v7.l
	v_add_nc_u16 v7.l, 0xe000, v79.l
	v_add_nc_u16 v144.h, 0xe000, v9.l
	v_add_nc_u16 v9.l, 0xe000, v80.l
	v_add_nc_u16 v79.l, 0xe000, v81.l
	v_add_nc_u16 v81.l, 0xe000, v82.l
	v_add_nc_u16 v83.h, 0xe000, v83.h
	v_add_nc_u16 v84.h, 0xe000, v84.h
	v_lshlrev_b32_e32 v157, 4, v115
	v_and_b32_e32 v114, 0xf0f0f0f, v114
	v_add_nc_u16 v85.l, 0xe000, v85.l
	v_add_nc_u16 v88.h, 0xe000, v88.h
	v_add_nc_u16 v106.l, 0xe000, v106.l
	v_add_nc_u16 v107.h, 0xe000, v107.h
	v_add_nc_u16 v108.l, 0xe000, v108.l
	v_add_nc_u16 v109.h, 0xe000, v109.h
	v_and_or_b32 v150, 0x30303030, v154, v136
	v_and_or_b32 v154, 0x30303030, v156, v118
	v_lshlrev_b16 v10.l, 8, v122.l
	v_and_b16 v79.h, 0x3f00, v122.h
	v_and_b16 v80.h, 0x3f00, v123.l
	v_and_b16 v82.h, 0x3f00, v124.l
	v_lshlrev_b16 v83.l, 8, v124.h
	v_and_b16 v85.h, 0x3f00, v125.l
	v_lshlrev_b16 v86.l, 8, v126.h
	v_and_b16 v86.h, 0x3f00, v125.h
	v_or_b16 v6.l, v6.h, v6.l
	v_or_b16 v8.l, v8.h, v8.l
	v_lshrrev_b16 v7.l, 8, v7.l
	v_lshrrev_b16 v9.l, 8, v9.l
	v_lshrrev_b16 v81.l, 8, v81.l
	v_lshrrev_b16 v83.h, 8, v83.h
	v_lshrrev_b16 v84.h, 8, v84.h
	v_and_or_b32 v156, 0x30303030, v157, v139
	v_and_or_b32 v157, 0x30303030, v115, v114
	v_lshlrev_b16 v87.l, 8, v127.l
	v_and_b16 v87.h, 0x3f00, v126.l
	v_and_b16 v107.l, 0x3f00, v127.h
	v_and_b16 v108.h, 0x3f00, v128.l
	v_and_b16 v109.l, 0x3f00, v128.h
	v_lshlrev_b16 v110.l, 8, v130.l
	v_and_b16 v110.h, 0x3f00, v129.l
	v_and_b16 v111.l, 0x3f00, v129.h
	v_lshlrev_b16 v111.h, 8, v130.h
	v_lshrrev_b16 v85.l, 8, v85.l
	v_lshrrev_b16 v88.h, 8, v88.h
	v_lshrrev_b16 v106.l, 8, v106.l
	v_lshrrev_b16 v107.h, 8, v107.h
	v_lshrrev_b16 v108.l, 8, v108.l
	v_lshrrev_b16 v109.h, 8, v109.h
	v_lshlrev_b16 v112.l, 8, v140.l
	v_lshlrev_b16 v113.l, 8, v140.h
	v_lshlrev_b16 v114.l, 8, v141.l
	v_lshlrev_b16 v115.h, 8, v141.h
	v_lshlrev_b16 v116.l, 8, v143.l
	v_and_b16 v118.h, 0x3f00, v143.l
	v_lshlrev_b16 v119.h, 8, v131.h
	v_lshlrev_b16 v120.l, 8, v144.l
	v_and_b16 v122.h, 0x3f00, v144.l
	v_lshlrev_b16 v128.l, 8, v146.l
	v_lshlrev_b16 v136.l, 8, v149.l
	v_lshlrev_b16 v138.l, 8, v150.l
	v_lshlrev_b16 v7.h, 8, v150.h
	v_lshlrev_b16 v8.h, 8, v151.l
	v_add_nc_u16 v10.l, 0xe000, v10.l
	v_add_nc_u16 v143.l, 0xe000, v6.l
	v_lshlrev_b16 v6.l, 8, v151.h
	v_add_nc_u16 v144.l, 0xe000, v8.l
	v_lshlrev_b16 v8.l, 8, v152.l
	v_or_b16 v7.l, v79.h, v7.l
	v_lshlrev_b16 v79.h, 8, v152.h
	v_or_b16 v9.l, v80.h, v9.l
	v_lshlrev_b16 v80.h, 8, v153.l
	v_add_nc_u16 v82.l, 0xe000, v83.l
	v_or_b16 v81.l, v82.h, v81.l
	v_lshlrev_b16 v82.h, 8, v153.h
	v_or_b16 v83.h, v85.h, v83.h
	v_lshlrev_b16 v85.h, 8, v154.l
	v_add_nc_u16 v86.l, 0xe000, v86.l
	v_or_b16 v84.h, v86.h, v84.h
	v_lshlrev_b16 v86.h, 8, v154.h
	v_and_b16 v10.h, 0x3f00, v122.l
	v_and_b16 v84.l, 0x3f00, v124.h
	v_and_b16 v112.h, 0x3f00, v130.l
	v_lshlrev_b16 v118.l, 8, v131.l
	v_and_b16 v121.l, 0x3f00, v131.h
	v_lshlrev_b16 v122.l, 8, v132.l
	v_lshlrev_b16 v123.l, 8, v132.h
	v_lshlrev_b16 v124.l, 8, v145.l
	v_and_b16 v124.h, 0x3f00, v132.l
	v_and_b16 v125.l, 0x3f00, v132.h
	v_lshlrev_b16 v125.h, 8, v145.h
	v_lshlrev_b16 v126.l, 8, v133.l
	v_lshlrev_b16 v127.h, 8, v133.h
	v_and_b16 v128.h, 0x3f00, v133.l
	v_lshlrev_b16 v129.h, 8, v146.h
	v_lshlrev_b16 v130.l, 8, v134.l
	v_lshlrev_b16 v131.h, 8, v134.h
	v_lshlrev_b16 v132.l, 8, v147.l
	v_and_b16 v132.h, 0x3f00, v134.l
	v_lshlrev_b16 v133.l, 8, v147.h
	v_lshlrev_b16 v134.l, 8, v148.l
	v_lshlrev_b16 v135.h, 8, v148.h
	v_lshlrev_b16 v137.h, 8, v149.h
	v_add_nc_u16 v87.l, 0xe000, v87.l
	v_or_b16 v85.l, v87.h, v85.l
	v_lshlrev_b16 v87.h, 8, v155.l
	v_or_b16 v88.h, v107.l, v88.h
	v_lshlrev_b16 v107.l, 8, v155.h
	v_or_b16 v106.l, v108.h, v106.l
	v_lshlrev_b16 v108.h, 8, v156.l
	v_add_nc_u16 v110.l, 0xe000, v110.l
	v_or_b16 v107.h, v109.l, v107.h
	v_lshlrev_b16 v109.l, 8, v156.h
	v_add_nc_u16 v111.h, 0xe000, v111.h
	v_or_b16 v108.l, v110.h, v108.l
	v_lshlrev_b16 v110.h, 8, v157.l
	v_or_b16 v109.h, v111.l, v109.h
	v_lshlrev_b16 v111.l, 8, v157.h
	v_and_b16 v88.l, 0x3f00, v126.h
	v_lshrrev_b16 v10.l, 8, v10.l
	v_lshrrev_b16 v79.l, 8, v79.l
	v_lshrrev_b16 v82.l, 8, v82.l
	v_lshrrev_b16 v86.l, 8, v86.l
	v_add_nc_u16 v112.l, 0xe000, v112.l
	v_add_nc_u16 v113.l, 0xe000, v113.l
	v_add_nc_u16 v114.l, 0xe000, v114.l
	v_add_nc_u16 v115.h, 0xe000, v115.h
	v_add_nc_u16 v119.h, 0xe000, v119.h
	v_add_nc_u16 v120.l, 0xe000, v120.l
	v_add_nc_u16 v128.l, 0xe000, v128.l
	v_add_nc_u16 v136.l, 0xe000, v136.l
	v_add_nc_u16 v138.l, 0xe000, v138.l
	v_add_nc_u16 v7.h, 0xe000, v7.h
	v_add_nc_u16 v8.h, 0xe000, v8.h
	v_add_nc_u16 v6.l, 0xe000, v6.l
	v_add_nc_u16 v8.l, 0xe000, v8.l
	v_add_nc_u16 v79.h, 0xe000, v79.h
	v_add_nc_u16 v80.h, 0xe000, v80.h
	v_add_nc_u16 v82.h, 0xe000, v82.h
	v_add_nc_u16 v85.h, 0xe000, v85.h
	v_add_nc_u16 v86.h, 0xe000, v86.h
	v_and_b16 v106.h, 0x3f00, v127.l
	v_and_b16 v113.h, 0x3f00, v130.h
	v_lshrrev_b16 v87.l, 8, v87.l
	v_lshrrev_b16 v110.l, 8, v110.l
	v_lshrrev_b16 v111.h, 8, v111.h
	v_add_nc_u16 v116.l, 0xe000, v116.l
	v_add_nc_u16 v117.h, 0xe000, v117.h
	v_add_nc_u16 v118.l, 0xe000, v118.l
	v_add_nc_u16 v121.h, 0xe000, v121.h
	v_add_nc_u16 v122.l, 0xe000, v122.l
	v_add_nc_u16 v123.l, 0xe000, v123.l
	v_add_nc_u16 v124.l, 0xe000, v124.l
	v_add_nc_u16 v125.h, 0xe000, v125.h
	v_add_nc_u16 v126.l, 0xe000, v126.l
	v_add_nc_u16 v127.h, 0xe000, v127.h
	v_add_nc_u16 v129.h, 0xe000, v129.h
	v_add_nc_u16 v130.l, 0xe000, v130.l
	v_add_nc_u16 v131.h, 0xe000, v131.h
	v_add_nc_u16 v132.l, 0xe000, v132.l
	v_add_nc_u16 v133.l, 0xe000, v133.l
	v_add_nc_u16 v134.l, 0xe000, v134.l
	v_add_nc_u16 v135.h, 0xe000, v135.h
	v_add_nc_u16 v137.h, 0xe000, v137.h
	v_add_nc_u16 v87.h, 0xe000, v87.h
	v_add_nc_u16 v107.l, 0xe000, v107.l
	v_add_nc_u16 v108.h, 0xe000, v108.h
	v_add_nc_u16 v109.l, 0xe000, v109.l
	v_add_nc_u16 v110.h, 0xe000, v110.h
	v_add_nc_u16 v111.l, 0xe000, v111.l
	v_and_b16 v114.h, 0x3f00, v140.l
	v_and_b16 v115.l, 0x3f00, v140.h
	v_and_b16 v116.h, 0x3f00, v141.l
	v_and_b16 v117.l, 0x3f00, v141.h
	v_and_b16 v130.h, 0x3f00, v146.l
	v_and_b16 v6.h, 0x3f00, v149.l
	v_and_b16 v9.h, 0x3f00, v150.l
	v_and_b16 v139.l, 0x3f00, v150.h
	v_or_b16 v10.l, v10.h, v10.l
	v_and_b16 v10.h, 0x3f00, v151.l
	v_and_b16 v80.l, 0x3f00, v151.h
	v_or_b16 v79.l, v81.h, v79.l
	v_and_b16 v81.h, 0x3f00, v152.l
	v_and_b16 v83.l, 0x3f00, v152.h
	v_or_b16 v82.l, v84.l, v82.l
	v_and_b16 v84.l, 0x3f00, v153.l
	v_and_b16 v139.h, 0x3f00, v153.h
	v_or_b16 v86.l, v88.l, v86.l
	v_and_b16 v88.l, 0x3f00, v154.l
	v_and_b16 v140.l, 0x3f00, v154.h
	v_lshrrev_b16 v112.l, 8, v112.l
	v_lshrrev_b16 v113.l, 8, v113.l
	v_lshrrev_b16 v114.l, 8, v114.l
	v_lshrrev_b16 v115.h, 8, v115.h
	v_lshrrev_b16 v119.h, 8, v119.h
	v_lshrrev_b16 v120.l, 8, v120.l
	v_lshrrev_b16 v128.l, 8, v128.l
	v_lshrrev_b16 v136.l, 8, v136.l
	v_lshrrev_b16 v138.l, 8, v138.l
	v_lshrrev_b16 v7.h, 8, v7.h
	v_lshrrev_b16 v8.h, 8, v8.h
	v_lshrrev_b16 v6.l, 8, v6.l
	v_lshrrev_b16 v8.l, 8, v8.l
	v_lshrrev_b16 v79.h, 8, v79.h
	v_lshrrev_b16 v80.h, 8, v80.h
	v_lshrrev_b16 v82.h, 8, v82.h
	v_lshrrev_b16 v85.h, 8, v85.h
	v_lshrrev_b16 v86.h, 8, v86.h
	v_and_b16 v120.h, 0x3f00, v131.l
	v_and_b16 v126.h, 0x3f00, v145.l
	v_and_b16 v127.l, 0x3f00, v145.h
	v_and_b16 v129.l, 0x3f00, v133.h
	v_and_b16 v131.l, 0x3f00, v146.h
	v_and_b16 v133.h, 0x3f00, v134.h
	v_and_b16 v134.h, 0x3f00, v147.l
	v_and_b16 v135.l, 0x3f00, v147.h
	v_and_b16 v136.h, 0x3f00, v148.l
	v_and_b16 v137.l, 0x3f00, v148.h
	v_and_b16 v138.h, 0x3f00, v149.h
	v_or_b16 v87.l, v106.h, v87.l
	v_and_b16 v106.h, 0x3f00, v155.l
	v_and_b16 v140.h, 0x3f00, v155.h
	v_or_b16 v110.l, v112.h, v110.l
	v_and_b16 v112.h, 0x3f00, v156.l
	v_and_b16 v141.l, 0x3f00, v156.h
	v_or_b16 v111.h, v113.h, v111.h
	v_and_b16 v113.h, 0x3f00, v157.l
	v_and_b16 v141.h, 0x3f00, v157.h
	v_lshrrev_b16 v116.l, 8, v116.l
	v_lshrrev_b16 v117.h, 8, v117.h
	v_lshrrev_b16 v118.l, 8, v118.l
	v_lshrrev_b16 v121.h, 8, v121.h
	v_lshrrev_b16 v122.l, 8, v122.l
	v_lshrrev_b16 v123.l, 8, v123.l
	v_lshrrev_b16 v124.l, 8, v124.l
	v_lshrrev_b16 v125.h, 8, v125.h
	v_lshrrev_b16 v126.l, 8, v126.l
	v_lshrrev_b16 v127.h, 8, v127.h
	v_lshrrev_b16 v129.h, 8, v129.h
	v_lshrrev_b16 v130.l, 8, v130.l
	v_lshrrev_b16 v131.h, 8, v131.h
	v_lshrrev_b16 v132.l, 8, v132.l
	v_lshrrev_b16 v133.l, 8, v133.l
	v_lshrrev_b16 v134.l, 8, v134.l
	v_lshrrev_b16 v135.h, 8, v135.h
	v_lshrrev_b16 v137.h, 8, v137.h
	v_lshrrev_b16 v87.h, 8, v87.h
	v_lshrrev_b16 v107.l, 8, v107.l
	v_lshrrev_b16 v108.h, 8, v108.h
	v_lshrrev_b16 v109.l, 8, v109.l
	v_lshrrev_b16 v110.h, 8, v110.h
	v_lshrrev_b16 v111.l, 8, v111.l
	v_or_b16 v112.l, v114.h, v112.l
	v_or_b16 v113.l, v115.l, v113.l
	v_or_b16 v114.l, v116.h, v114.l
	v_or_b16 v114.h, v117.l, v115.h
	v_or_b16 v116.h, v121.l, v119.h
	v_or_b16 v117.l, v122.h, v120.l
	v_or_b16 v121.l, v130.h, v128.l
	v_or_b16 v6.h, v6.h, v136.l
	v_or_b16 v9.h, v9.h, v138.l
	v_or_b16 v7.h, v139.l, v7.h
	v_or_b16 v8.h, v10.h, v8.h
	v_or_b16 v6.l, v80.l, v6.l
	v_or_b16 v8.l, v81.h, v8.l
	v_or_b16 v10.h, v83.l, v79.h
	v_or_b16 v79.h, v84.l, v80.h
	v_or_b16 v80.l, v139.h, v82.h
	v_or_b16 v80.h, v88.l, v85.h
	v_or_b16 v81.h, v140.l, v86.h
	v_or_b16 v115.l, v118.h, v116.l
	v_or_b16 v115.h, v119.l, v117.h
	v_or_b16 v116.l, v120.h, v118.l
	v_or_b16 v117.h, v123.h, v121.h
	v_or_b16 v118.l, v124.h, v122.l
	v_or_b16 v118.h, v125.l, v123.l
	v_or_b16 v119.l, v126.h, v124.l
	v_or_b16 v119.h, v127.l, v125.h
	v_or_b16 v120.l, v128.h, v126.l
	v_or_b16 v120.h, v129.l, v127.h
	v_or_b16 v121.h, v131.l, v129.h
	v_or_b16 v122.l, v132.h, v130.l
	v_or_b16 v122.h, v133.h, v131.h
	v_or_b16 v123.l, v134.h, v132.l
	v_or_b16 v123.h, v135.l, v133.l
	v_or_b16 v124.l, v136.h, v134.l
	v_or_b16 v124.h, v137.l, v135.h
	v_or_b16 v125.l, v138.h, v137.h
	v_or_b16 v82.h, v106.h, v87.h
	v_or_b16 v83.l, v140.h, v107.l
	v_or_b16 v84.l, v112.h, v108.h
	v_or_b16 v85.h, v141.l, v109.l
	v_or_b16 v86.h, v113.h, v110.h
	v_or_b16 v87.h, v141.h, v111.l
	v_add_nc_u16 v126.l, 0xe000, v10.l
	v_add_nc_u16 v126.h, 0xe000, v7.l
	v_add_nc_u16 v127.l, 0xe000, v9.l
	v_add_nc_u16 v127.h, 0xe000, v79.l
	v_add_nc_u16 v128.l, 0xe000, v81.l
	v_add_nc_u16 v128.h, 0xe000, v82.l
	v_add_nc_u16 v129.l, 0xe000, v83.h
	v_add_nc_u16 v129.h, 0xe000, v84.h
	v_add_nc_u16 v130.l, 0xe000, v85.l
	v_add_nc_u16 v130.h, 0xe000, v86.l
	v_add_nc_u16 v88.l, 0xe000, v87.l
	v_add_nc_u16 v88.h, 0xe000, v88.h
	v_add_nc_u16 v106.l, 0xe000, v106.l
	v_add_nc_u16 v106.h, 0xe000, v107.h
	v_add_nc_u16 v107.l, 0xe000, v108.l
	v_add_nc_u16 v107.h, 0xe000, v109.h
	v_add_nc_u16 v108.l, 0xe000, v110.l
	v_add_nc_u16 v108.h, 0xe000, v111.h
	v_add_nc_u16 v109.l, 0xe000, v112.l
	v_add_nc_u16 v109.h, 0xe000, v113.l
	v_add_nc_u16 v113.l, 0xe000, v117.l
	v_add_nc_u16 v117.l, 0xe000, v121.l
	v_add_nc_u16 v121.l, 0xe000, v6.h
	v_add_nc_u16 v7.l, 0xe000, v9.h
	v_add_nc_u16 v7.h, 0xe000, v7.h
	v_add_nc_u16 v9.l, 0xe000, v8.h
	v_add_nc_u16 v9.h, 0xe000, v6.l
	v_add_nc_u16 v6.l, 0xe000, v8.l
	v_add_nc_u16 v6.h, 0xe000, v10.h
	v_add_nc_u16 v8.l, 0xe000, v79.h
	v_add_nc_u16 v8.h, 0xe000, v80.l
	v_add_nc_u16 v10.l, 0xe000, v80.h
	v_add_nc_u16 v10.h, 0xe000, v81.h
	ds_store_2addr_b32 v59, v143, v144 offset0:64 offset1:80
	v_add_nc_u16 v110.l, 0xe000, v114.l
	v_add_nc_u16 v110.h, 0xe000, v114.h
	v_add_nc_u16 v111.l, 0xe000, v115.l
	v_add_nc_u16 v111.h, 0xe000, v115.h
	v_add_nc_u16 v112.l, 0xe000, v116.l
	v_add_nc_u16 v112.h, 0xe000, v116.h
	v_add_nc_u16 v113.h, 0xe000, v117.h
	v_add_nc_u16 v114.l, 0xe000, v118.l
	v_add_nc_u16 v114.h, 0xe000, v118.h
	v_add_nc_u16 v115.l, 0xe000, v119.l
	v_add_nc_u16 v115.h, 0xe000, v119.h
	v_add_nc_u16 v116.l, 0xe000, v120.l
	v_add_nc_u16 v116.h, 0xe000, v120.h
	v_add_nc_u16 v117.h, 0xe000, v121.h
	v_add_nc_u16 v118.l, 0xe000, v122.l
	v_add_nc_u16 v118.h, 0xe000, v122.h
	v_add_nc_u16 v119.l, 0xe000, v123.l
	v_add_nc_u16 v119.h, 0xe000, v123.h
	v_add_nc_u16 v120.l, 0xe000, v124.l
	v_add_nc_u16 v120.h, 0xe000, v124.h
	v_add_nc_u16 v121.h, 0xe000, v125.l
	v_add_nc_u16 v79.l, 0xe000, v82.h
	v_add_nc_u16 v79.h, 0xe000, v83.l
	v_add_nc_u16 v80.l, 0xe000, v84.l
	v_add_nc_u16 v80.h, 0xe000, v85.h
	v_add_nc_u16 v81.l, 0xe000, v86.h
	v_add_nc_u16 v81.h, 0xe000, v87.h
	ds_store_2addr_b32 v60, v126, v127 offset0:112 offset1:128
	ds_store_2addr_b32 v61, v128, v129 offset0:160 offset1:176
	ds_store_2addr_b32 v62, v130, v88 offset0:208 offset1:224
	ds_store_2addr_b32 v63, v106, v107 offset1:16
	ds_store_2addr_b32 v64, v108, v109 offset0:48 offset1:64
	ds_store_2addr_b32 v65, v110, v111 offset0:96 offset1:112
	ds_store_2addr_b32 v66, v112, v113 offset0:144 offset1:160
	ds_store_2addr_b32 v67, v114, v115 offset0:192 offset1:208
	ds_store_2addr_b32 v68, v116, v117 offset0:112 offset1:128
	ds_store_2addr_b32 v69, v118, v119 offset0:32 offset1:48
	ds_store_2addr_b32 v70, v120, v121 offset0:80 offset1:96
	ds_store_2addr_b32 v71, v7, v9 offset0:128 offset1:144
	ds_store_2addr_b32 v72, v6, v8 offset0:176 offset1:192
	ds_store_2addr_b32 v73, v10, v79 offset0:224 offset1:240
	ds_store_2addr_b32 v74, v80, v81 offset0:16 offset1:32
	ds_store_b32 v55, v142 offset:9728
	s_waitcnt vmcnt(19)
	ds_store_b32 v57, v3 offset:9732
	s_waitcnt vmcnt(18)
	ds_store_b32 v58, v4 offset:9732
	s_waitcnt vmcnt(16)
	ds_store_2addr_stride64_b32 v52, v5, v89 offset0:1 offset1:3
	s_waitcnt vmcnt(14)
	ds_store_2addr_stride64_b32 v52, v90, v91 offset0:5 offset1:7
	s_waitcnt vmcnt(12)
	ds_store_2addr_stride64_b32 v52, v92, v93 offset0:9 offset1:11
	s_waitcnt vmcnt(10)
	ds_store_2addr_stride64_b32 v52, v94, v95 offset0:13 offset1:15
	s_waitcnt vmcnt(7)
	ds_store_2addr_stride64_b32 v52, v97, v98 offset0:17 offset1:19
	s_waitcnt vmcnt(5)
	ds_store_2addr_stride64_b32 v52, v99, v100 offset0:21 offset1:23
	s_waitcnt vmcnt(3)
	ds_store_2addr_stride64_b32 v52, v101, v102 offset0:25 offset1:27
	s_waitcnt vmcnt(1)
	ds_store_2addr_stride64_b32 v52, v103, v104 offset0:29 offset1:31
	s_waitcnt vmcnt(0)
	ds_store_2addr_stride64_b32 v52, v96, v105 offset0:33 offset1:35
	v_dual_mov_b32 v79, v51 :: v_dual_mov_b32 v80, v49
	v_mov_b32_e32 v81, v50
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_2addr_b32 v[3:4], v75 offset1:152
	ds_load_2addr_b32 v[5:6], v76 offset0:48 offset1:200
	ds_load_2addr_b32 v[7:8], v77 offset0:96 offset1:248
	ds_load_2addr_b32 v[9:10], v78 offset0:16 offset1:168
.LBB0_2:                                ; %.preheader8.i.i.i
                                        ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v89, s11 :: v_dual_add_nc_u32 v90, 0x2500, v80
	v_dual_mov_b32 v82, s4 :: v_dual_add_nc_u32 v91, 0xa10, v81
	v_add_nc_u32_e32 v92, 0x1310, v81
	v_add_nc_u32_e32 v93, 0x1c10, v81
	ds_load_2addr_b64 v[98:101], v81 offset0:34 offset1:35
	ds_load_2addr_b64 v[122:125], v90 offset1:1
	ds_load_2addr_b64 v[106:109], v91 offset1:1
	ds_load_2addr_b64 v[114:117], v92 offset1:1
	ds_load_2addr_b64 v[126:129], v93 offset1:1
	ds_load_i8 v134, v79 offset:9732
	ds_load_i8 v135, v79 offset:10340
	ds_load_i8 v136, v79 offset:10948
	ds_load_i8 v137, v79 offset:11556
	ds_load_i8 v138, v79 offset:12164
	ds_load_i8 v139, v79 offset:12772
	ds_load_i8 v140, v79 offset:13380
	ds_load_i8 v141, v79 offset:13988
	s_add_i32 s18, s18, 4
	v_dual_mov_b32 v88, s10 :: v_dual_mov_b32 v87, s9
	s_lshr_b32 s19, s18, 1
	v_dual_mov_b32 v86, s8 :: v_dual_mov_b32 v85, s7
	s_and_b32 s19, s19, 0x7ffffffc
	v_dual_mov_b32 v84, s6 :: v_dual_mov_b32 v83, s5
	v_add_nc_u32_e32 v90, s19, v50
	ds_load_2addr_stride64_b32 v[130:131], v90 offset0:1 offset1:10
	ds_load_2addr_stride64_b32 v[132:133], v90 offset0:19 offset1:28
	s_cmp_lt_u32 s18, 28
	s_waitcnt lgkmcnt(13)
	v_wmma_i32_16x16x16_iu8 v[90:97], v[122:125], v[98:101], v[82:89] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu8 v[98:105], v[122:125], v[106:109], v[82:89] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu8 v[106:113], v[122:125], v[114:117], v[82:89] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu8 v[114:121], v[122:125], v[126:129], v[82:89] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(9)
	v_mul_lo_u32 v82, v90, v134
	s_waitcnt lgkmcnt(8)
	v_mul_lo_u32 v83, v91, v135
	s_waitcnt lgkmcnt(7)
	v_mul_lo_u32 v84, v92, v136
	s_waitcnt lgkmcnt(6)
	v_mul_lo_u32 v85, v93, v137
	s_waitcnt lgkmcnt(5)
	v_mul_lo_u32 v86, v94, v138
	s_waitcnt lgkmcnt(4)
	v_mul_lo_u32 v87, v95, v139
	s_waitcnt lgkmcnt(3)
	v_mul_lo_u32 v88, v96, v140
	s_waitcnt lgkmcnt(2)
	v_mul_lo_u32 v89, v97, v141
	v_mul_lo_u32 v90, v98, v134
	v_mul_lo_u32 v91, v99, v135
	v_mul_lo_u32 v92, v100, v136
	v_mul_lo_u32 v93, v101, v137
	v_mul_lo_u32 v94, v102, v138
	v_mul_lo_u32 v95, v103, v139
	v_mul_lo_u32 v96, v104, v140
	v_mul_lo_u32 v97, v105, v141
	v_mul_lo_u32 v98, v106, v134
	v_mul_lo_u32 v99, v107, v135
	v_mul_lo_u32 v100, v108, v136
	v_mul_lo_u32 v101, v109, v137
	v_mul_lo_u32 v102, v110, v138
	v_mul_lo_u32 v103, v111, v139
	v_mul_lo_u32 v104, v112, v140
	v_mul_lo_u32 v105, v113, v141
	v_mul_lo_u32 v106, v114, v134
	v_mul_lo_u32 v107, v115, v135
	v_mul_lo_u32 v108, v116, v136
	v_mul_lo_u32 v109, v117, v137
	v_mul_lo_u32 v110, v118, v138
	v_mul_lo_u32 v111, v119, v139
	v_mul_lo_u32 v112, v120, v140
	v_mul_lo_u32 v113, v121, v141
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v83, v83
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v86, v86
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v88, v88
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v91, v91
	v_cvt_f32_i32_e32 v92, v92
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v96, v96
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v99, v99
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v102, v102
	v_cvt_f32_i32_e32 v103, v103
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_cvt_f32_i32_e32 v112, v112
	v_cvt_f32_i32_e32 v113, v113
	v_dual_mul_f32 v84, v5, v84 :: v_dual_add_nc_u32 v81, 16, v81
	v_dual_mul_f32 v85, v6, v85 :: v_dual_add_nc_u32 v80, 16, v80
	v_dual_mul_f32 v86, v7, v86 :: v_dual_add_nc_u32 v79, 1, v79
	v_dual_mul_f32 v82, v3, v82 :: v_dual_mul_f32 v83, v4, v83
	v_dual_mul_f32 v87, v8, v87 :: v_dual_mul_f32 v88, v9, v88
	v_dual_mul_f32 v89, v10, v89 :: v_dual_mul_f32 v90, v3, v90
	v_dual_mul_f32 v91, v4, v91 :: v_dual_mul_f32 v92, v5, v92
	v_dual_mul_f32 v93, v6, v93 :: v_dual_mul_f32 v94, v7, v94
	v_dual_mul_f32 v95, v8, v95 :: v_dual_mul_f32 v96, v9, v96
	v_dual_mul_f32 v97, v10, v97 :: v_dual_mul_f32 v98, v3, v98
	v_dual_mul_f32 v99, v4, v99 :: v_dual_mul_f32 v100, v5, v100
	v_dual_mul_f32 v101, v6, v101 :: v_dual_mul_f32 v102, v7, v102
	v_dual_mul_f32 v103, v8, v103 :: v_dual_mul_f32 v104, v9, v104
	v_dual_mul_f32 v105, v10, v105 :: v_dual_mul_f32 v106, v3, v106
	v_dual_mul_f32 v107, v4, v107 :: v_dual_mul_f32 v108, v5, v108
	v_dual_mul_f32 v109, v6, v109 :: v_dual_mul_f32 v110, v7, v110
	v_dual_mul_f32 v111, v8, v111 :: v_dual_mul_f32 v112, v9, v112
	v_mul_f32_e32 v113, v10, v113
	s_waitcnt lgkmcnt(1)
	v_dual_fmac_f32 v32, v130, v82 :: v_dual_fmac_f32 v35, v131, v91
	v_dual_fmac_f32 v46, v130, v83 :: v_dual_fmac_f32 v33, v131, v92
	v_dual_fmac_f32 v44, v130, v84 :: v_dual_fmac_f32 v31, v131, v93
	v_dual_fmac_f32 v43, v130, v85 :: v_dual_fmac_f32 v36, v131, v90
	v_dual_fmac_f32 v40, v130, v86 :: v_dual_fmac_f32 v29, v131, v95
	v_dual_fmac_f32 v39, v130, v87 :: v_dual_fmac_f32 v30, v131, v94
	v_dual_fmac_f32 v38, v130, v88 :: v_dual_fmac_f32 v27, v131, v97
	v_dual_fmac_f32 v37, v130, v89 :: v_dual_fmac_f32 v28, v131, v96
	s_waitcnt lgkmcnt(0)
	v_dual_fmac_f32 v26, v132, v98 :: v_dual_fmac_f32 v17, v133, v107
	v_dual_fmac_f32 v25, v132, v99 :: v_dual_fmac_f32 v18, v133, v106
	v_dual_fmac_f32 v24, v132, v100 :: v_dual_fmac_f32 v15, v133, v109
	v_dual_fmac_f32 v23, v132, v101 :: v_dual_fmac_f32 v16, v133, v108
	v_dual_fmac_f32 v22, v132, v102 :: v_dual_fmac_f32 v13, v133, v111
	v_dual_fmac_f32 v21, v132, v103 :: v_dual_fmac_f32 v14, v133, v110
	v_dual_fmac_f32 v20, v132, v104 :: v_dual_fmac_f32 v11, v133, v113
	v_dual_fmac_f32 v19, v132, v105 :: v_dual_fmac_f32 v12, v133, v112
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %_ZL18mmq_vec_dot_targetIL9ggml_type14ELi64ELb1ELb0EEvPKiS2_Pfi.exit.i
                                        ;   in Loop: Header=BB0_1 Depth=1
	v_add_nc_u32_e32 v2, s22, v2
	s_barrier
	buffer_gl0_inv
	s_mov_b32 s18, -4
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_u32 v2, vcc_lo, s14, v2
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	s_clause 0x7
	global_load_b32 v8, v[2:3], off
	global_load_b32 v9, v[2:3], off offset:512
	global_load_b32 v10, v[2:3], off offset:1024
	global_load_b32 v79, v[2:3], off offset:1536
	global_load_b32 v80, v[2:3], off offset:2048
	global_load_b32 v81, v[2:3], off offset:2560
	global_load_b32 v82, v[2:3], off offset:3072
	global_load_b32 v83, v[2:3], off offset:3584
	v_add_co_u32 v4, vcc_lo, 0x1000, v2
	v_add_co_ci_u32_e64 v5, null, 0, v3, vcc_lo
	v_add_co_u32 v6, vcc_lo, v2, 0x2000
	v_add_co_ci_u32_e64 v7, null, 0, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, 0x2000, v2
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0x9
	global_load_b32 v84, v[6:7], off offset:-4096
	global_load_b32 v6, v[6:7], off
	global_load_b32 v7, v[4:5], off offset:512
	global_load_b32 v85, v[4:5], off offset:1024
	global_load_b32 v86, v[4:5], off offset:1536
	global_load_b32 v87, v[4:5], off offset:2048
	global_load_b32 v88, v[4:5], off offset:2560
	global_load_b32 v89, v[4:5], off offset:3072
	global_load_b32 v4, v[4:5], off offset:3584
	global_load_b32 v2, v[2:3], off offset:512
	s_waitcnt vmcnt(16)
	ds_store_2addr_stride64_b32 v52, v8, v9 offset0:1 offset1:3
	s_waitcnt vmcnt(14)
	ds_store_2addr_stride64_b32 v52, v10, v79 offset0:5 offset1:7
	s_waitcnt vmcnt(12)
	ds_store_2addr_stride64_b32 v52, v80, v81 offset0:9 offset1:11
	s_waitcnt vmcnt(10)
	ds_store_2addr_stride64_b32 v52, v82, v83 offset0:13 offset1:15
	s_waitcnt vmcnt(7)
	ds_store_2addr_stride64_b32 v52, v84, v7 offset0:17 offset1:19
	s_waitcnt vmcnt(5)
	ds_store_2addr_stride64_b32 v52, v85, v86 offset0:21 offset1:23
	s_waitcnt vmcnt(3)
	ds_store_2addr_stride64_b32 v52, v87, v88 offset0:25 offset1:27
	s_waitcnt vmcnt(1)
	ds_store_2addr_stride64_b32 v52, v89, v4 offset0:29 offset1:31
	s_waitcnt vmcnt(0)
	ds_store_2addr_stride64_b32 v52, v6, v2 offset0:33 offset1:35
	v_dual_mov_b32 v10, v51 :: v_dual_mov_b32 v79, v49
	v_mov_b32_e32 v80, v50
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_2addr_b32 v[2:3], v75 offset1:152
	ds_load_2addr_b32 v[4:5], v76 offset0:48 offset1:200
	ds_load_2addr_b32 v[6:7], v77 offset0:96 offset1:248
	ds_load_2addr_b32 v[8:9], v78 offset0:16 offset1:168
.LBB0_4:                                ; %.preheader8.i.i89.i
                                        ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v88, s11 :: v_dual_add_nc_u32 v89, 0x2580, v79
	v_dual_mov_b32 v81, s4 :: v_dual_add_nc_u32 v90, 0xa10, v80
	v_add_nc_u32_e32 v91, 0x1310, v80
	v_add_nc_u32_e32 v92, 0x1c10, v80
	ds_load_2addr_b64 v[97:100], v80 offset0:34 offset1:35
	ds_load_2addr_b64 v[121:124], v89 offset1:1
	ds_load_2addr_b64 v[105:108], v90 offset1:1
	ds_load_2addr_b64 v[113:116], v91 offset1:1
	ds_load_2addr_b64 v[125:128], v92 offset1:1
	ds_load_i8 v133, v10 offset:9740
	ds_load_i8 v134, v10 offset:10348
	ds_load_i8 v135, v10 offset:10956
	ds_load_i8 v136, v10 offset:11564
	ds_load_i8 v137, v10 offset:12172
	ds_load_i8 v138, v10 offset:12780
	ds_load_i8 v139, v10 offset:13388
	ds_load_i8 v140, v10 offset:13996
	s_add_i32 s18, s18, 4
	v_dual_mov_b32 v87, s10 :: v_dual_mov_b32 v86, s9
	s_lshr_b32 s19, s18, 1
	v_dual_mov_b32 v85, s8 :: v_dual_mov_b32 v84, s7
	s_and_b32 s19, s19, 0x7ffffffc
	v_dual_mov_b32 v83, s6 :: v_dual_mov_b32 v82, s5
	v_add_nc_u32_e32 v89, s19, v50
	ds_load_2addr_stride64_b32 v[129:130], v89 offset0:1 offset1:10
	ds_load_2addr_stride64_b32 v[131:132], v89 offset0:19 offset1:28
	s_cmp_lt_u32 s18, 28
	s_waitcnt lgkmcnt(13)
	v_wmma_i32_16x16x16_iu8 v[89:96], v[121:124], v[97:100], v[81:88] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu8 v[97:104], v[121:124], v[105:108], v[81:88] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_wmma_i32_16x16x16_iu8 v[105:112], v[121:124], v[113:116], v[81:88] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(10)
	v_wmma_i32_16x16x16_iu8 v[113:120], v[121:124], v[125:128], v[81:88] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(9)
	v_mul_lo_u32 v81, v89, v133
	s_waitcnt lgkmcnt(8)
	v_mul_lo_u32 v82, v90, v134
	s_waitcnt lgkmcnt(7)
	v_mul_lo_u32 v83, v91, v135
	s_waitcnt lgkmcnt(6)
	v_mul_lo_u32 v84, v92, v136
	s_waitcnt lgkmcnt(5)
	v_mul_lo_u32 v85, v93, v137
	s_waitcnt lgkmcnt(4)
	v_mul_lo_u32 v86, v94, v138
	s_waitcnt lgkmcnt(3)
	v_mul_lo_u32 v87, v95, v139
	s_waitcnt lgkmcnt(2)
	v_mul_lo_u32 v88, v96, v140
	v_mul_lo_u32 v89, v97, v133
	v_mul_lo_u32 v90, v98, v134
	v_mul_lo_u32 v91, v99, v135
	v_mul_lo_u32 v92, v100, v136
	v_mul_lo_u32 v93, v101, v137
	v_mul_lo_u32 v94, v102, v138
	v_mul_lo_u32 v95, v103, v139
	v_mul_lo_u32 v96, v104, v140
	v_mul_lo_u32 v97, v105, v133
	v_mul_lo_u32 v98, v106, v134
	v_mul_lo_u32 v99, v107, v135
	v_mul_lo_u32 v100, v108, v136
	v_mul_lo_u32 v101, v109, v137
	v_mul_lo_u32 v102, v110, v138
	v_mul_lo_u32 v103, v111, v139
	v_mul_lo_u32 v104, v112, v140
	v_mul_lo_u32 v105, v113, v133
	v_mul_lo_u32 v106, v114, v134
	v_mul_lo_u32 v107, v115, v135
	v_mul_lo_u32 v108, v116, v136
	v_mul_lo_u32 v109, v117, v137
	v_mul_lo_u32 v110, v118, v138
	v_mul_lo_u32 v111, v119, v139
	v_mul_lo_u32 v112, v120, v140
	v_cvt_f32_i32_e32 v81, v81
	v_cvt_f32_i32_e32 v82, v82
	v_cvt_f32_i32_e32 v83, v83
	v_cvt_f32_i32_e32 v84, v84
	v_cvt_f32_i32_e32 v85, v85
	v_cvt_f32_i32_e32 v86, v86
	v_cvt_f32_i32_e32 v87, v87
	v_cvt_f32_i32_e32 v88, v88
	v_cvt_f32_i32_e32 v89, v89
	v_cvt_f32_i32_e32 v90, v90
	v_cvt_f32_i32_e32 v91, v91
	v_cvt_f32_i32_e32 v92, v92
	v_cvt_f32_i32_e32 v93, v93
	v_cvt_f32_i32_e32 v94, v94
	v_cvt_f32_i32_e32 v95, v95
	v_cvt_f32_i32_e32 v96, v96
	v_cvt_f32_i32_e32 v97, v97
	v_cvt_f32_i32_e32 v98, v98
	v_cvt_f32_i32_e32 v99, v99
	v_cvt_f32_i32_e32 v100, v100
	v_cvt_f32_i32_e32 v101, v101
	v_cvt_f32_i32_e32 v102, v102
	v_cvt_f32_i32_e32 v103, v103
	v_cvt_f32_i32_e32 v104, v104
	v_cvt_f32_i32_e32 v105, v105
	v_cvt_f32_i32_e32 v106, v106
	v_cvt_f32_i32_e32 v107, v107
	v_cvt_f32_i32_e32 v108, v108
	v_cvt_f32_i32_e32 v109, v109
	v_cvt_f32_i32_e32 v110, v110
	v_cvt_f32_i32_e32 v111, v111
	v_cvt_f32_i32_e32 v112, v112
	v_dual_mul_f32 v83, v4, v83 :: v_dual_add_nc_u32 v80, 16, v80
	v_dual_mul_f32 v84, v5, v84 :: v_dual_add_nc_u32 v79, 16, v79
	v_dual_mul_f32 v85, v6, v85 :: v_dual_add_nc_u32 v10, 1, v10
	v_dual_mul_f32 v81, v2, v81 :: v_dual_mul_f32 v82, v3, v82
	v_dual_mul_f32 v86, v7, v86 :: v_dual_mul_f32 v87, v8, v87
	v_dual_mul_f32 v88, v9, v88 :: v_dual_mul_f32 v89, v2, v89
	v_dual_mul_f32 v90, v3, v90 :: v_dual_mul_f32 v91, v4, v91
	v_dual_mul_f32 v92, v5, v92 :: v_dual_mul_f32 v93, v6, v93
	v_dual_mul_f32 v94, v7, v94 :: v_dual_mul_f32 v95, v8, v95
	v_dual_mul_f32 v96, v9, v96 :: v_dual_mul_f32 v97, v2, v97
	v_dual_mul_f32 v98, v3, v98 :: v_dual_mul_f32 v99, v4, v99
	v_dual_mul_f32 v100, v5, v100 :: v_dual_mul_f32 v101, v6, v101
	v_dual_mul_f32 v102, v7, v102 :: v_dual_mul_f32 v103, v8, v103
	v_dual_mul_f32 v104, v9, v104 :: v_dual_mul_f32 v105, v2, v105
	v_dual_mul_f32 v106, v3, v106 :: v_dual_mul_f32 v107, v4, v107
	v_dual_mul_f32 v108, v5, v108 :: v_dual_mul_f32 v109, v6, v109
	v_dual_mul_f32 v110, v7, v110 :: v_dual_mul_f32 v111, v8, v111
	v_mul_f32_e32 v112, v9, v112
	s_waitcnt lgkmcnt(1)
	v_dual_fmac_f32 v32, v129, v81 :: v_dual_fmac_f32 v35, v130, v90
	v_dual_fmac_f32 v46, v129, v82 :: v_dual_fmac_f32 v33, v130, v91
	v_dual_fmac_f32 v44, v129, v83 :: v_dual_fmac_f32 v31, v130, v92
	v_dual_fmac_f32 v43, v129, v84 :: v_dual_fmac_f32 v36, v130, v89
	v_dual_fmac_f32 v40, v129, v85 :: v_dual_fmac_f32 v29, v130, v94
	v_dual_fmac_f32 v39, v129, v86 :: v_dual_fmac_f32 v30, v130, v93
	v_dual_fmac_f32 v38, v129, v87 :: v_dual_fmac_f32 v27, v130, v96
	v_dual_fmac_f32 v37, v129, v88 :: v_dual_fmac_f32 v28, v130, v95
	s_waitcnt lgkmcnt(0)
	v_dual_fmac_f32 v26, v131, v97 :: v_dual_fmac_f32 v17, v132, v106
	v_dual_fmac_f32 v25, v131, v98 :: v_dual_fmac_f32 v18, v132, v105
	v_dual_fmac_f32 v24, v131, v99 :: v_dual_fmac_f32 v15, v132, v108
	v_dual_fmac_f32 v23, v131, v100 :: v_dual_fmac_f32 v16, v132, v107
	v_dual_fmac_f32 v22, v131, v101 :: v_dual_fmac_f32 v13, v132, v110
	v_dual_fmac_f32 v21, v131, v102 :: v_dual_fmac_f32 v14, v132, v109
	v_dual_fmac_f32 v20, v131, v103 :: v_dual_fmac_f32 v11, v132, v112
	v_dual_fmac_f32 v19, v131, v104 :: v_dual_fmac_f32 v12, v132, v111
	s_cbranch_scc1 .LBB0_4
; %bb.5:                                ; %_ZL18mmq_vec_dot_targetIL9ggml_type14ELi64ELb1ELb0EEvPKiS2_Pfi.exit156.i
                                        ;   in Loop: Header=BB0_1 Depth=1
	s_add_i32 s23, s23, 1
	s_cmp_lg_u32 s23, 8
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_1
; %bb.6:                                ; %_ZL19dense_mmq_bf16_bodyIL9ggml_type14ELi64ELi8ELb1ELb1EEvPKcPKiP14__hip_bfloat16iiii.exit
	s_load_b32 s4, s[0:1], 0x18
	v_bfe_u32 v1, v32, 16, 1
	v_or_b32_e32 v2, 0x400000, v32
	v_bfe_u32 v3, v46, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v32, v32
	v_or_b32_e32 v4, 0x400000, v46
	v_add3_u32 v41, v1, v32, 0x7fff
	v_bfe_u32 v5, v44, 16, 1
	v_add3_u32 v3, v3, v46, 0x7fff
	s_lshl_b32 s0, s2, 6
	v_or_b32_e32 v6, 0x400000, v44
	v_cndmask_b32_e32 v32, v41, v2, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v46, v46
	v_bfe_u32 v7, v43, 16, 1
	v_add3_u32 v5, v5, v44, 0x7fff
	v_or_b32_e32 v8, 0x400000, v43
	v_bfe_u32 v9, v40, 16, 1
	v_bfe_u32 v10, v39, 16, 1
	v_add3_u32 v7, v7, v43, 0x7fff
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[0:1], null, s4, v34, v[0:1]
	s_mul_i32 s2, s4, s3
	v_cndmask_b32_e32 v34, v3, v4, vcc_lo
	s_ashr_i32 s3, s2, 31
	v_cmp_u_f32_e32 vcc_lo, v44, v44
	s_lshl_b64 s[2:3], s[2:3], 1
	v_add3_u32 v3, v9, v40, 0x7fff
	s_add_u32 s2, s16, s2
	v_ashrrev_i32_e32 v1, 31, v0
	s_addc_u32 s3, s17, s3
	s_ashr_i32 s1, s0, 31
	v_cndmask_b32_e32 v5, v5, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v43, v43
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	s_lshl_b64 s[0:1], s[0:1], 1
	v_or_b32_e32 v4, 0x400000, v40
	s_add_u32 s0, s2, s0
	v_cndmask_b32_e32 v6, v7, v8, vcc_lo
	s_addc_u32 s1, s3, s1
	v_add_co_u32 v1, vcc_lo, s0, v1
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v40, v40
	v_add3_u32 v7, v10, v39, 0x7fff
	v_or_b32_e32 v8, 0x400000, v39
	v_bfe_u32 v9, v38, 16, 1
	s_lshl_b32 s2, s4, 4
	v_cndmask_b32_e32 v10, v3, v4, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v39, v39
	v_bfe_u32 v3, v37, 16, 1
	v_add3_u32 v4, v9, v38, 0x7fff
	v_or_b32_e32 v39, 0x400000, v37
	v_bfe_u32 v40, v36, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_or_b32_e32 v8, 0x400000, v38
	v_cmp_u_f32_e32 vcc_lo, v38, v38
	v_add3_u32 v9, v3, v37, 0x7fff
	v_bfe_u32 v38, v35, 16, 1
	v_cndmask_b32_e32 v8, v4, v8, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v37, v37
	v_add_nc_u32_e32 v3, s2, v0
	v_or_b32_e32 v37, 0x400000, v36
	v_cndmask_b32_e32 v0, v9, v39, vcc_lo
	v_ashrrev_i32_e32 v4, 31, v3
	s_clause 0x7
	global_store_d16_hi_b16 v[1:2], v32, off
	global_store_d16_hi_b16 v[1:2], v34, off offset:4
	global_store_d16_hi_b16 v[1:2], v5, off offset:8
	global_store_d16_hi_b16 v[1:2], v6, off offset:12
	global_store_d16_hi_b16 v[1:2], v10, off offset:16
	global_store_d16_hi_b16 v[1:2], v7, off offset:20
	global_store_d16_hi_b16 v[1:2], v8, off offset:24
	global_store_d16_hi_b16 v[1:2], v0, off offset:28
	v_add3_u32 v9, v40, v36, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v36, v36
	v_bfe_u32 v7, v31, 16, 1
	v_lshlrev_b64 v[0:1], 1, v[3:4]
	v_or_b32_e32 v8, 0x400000, v31
	v_add3_u32 v2, v38, v35, 0x7fff
	v_or_b32_e32 v5, 0x400000, v35
	v_add3_u32 v7, v7, v31, 0x7fff
	v_cndmask_b32_e32 v4, v9, v37, vcc_lo
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_bfe_u32 v6, v33, 16, 1
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v35, v35
	v_bfe_u32 v9, v30, 16, 1
	v_or_b32_e32 v10, 0x400000, v29
	v_cndmask_b32_e32 v5, v2, v5, vcc_lo
	v_add3_u32 v2, v6, v33, 0x7fff
	v_or_b32_e32 v6, 0x400000, v33
	v_cmp_u_f32_e32 vcc_lo, v33, v33
	v_cndmask_b32_e32 v6, v2, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v31, v31
	v_bfe_u32 v2, v29, 16, 1
	v_bfe_u32 v31, v28, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v30, 0x7fff
	v_or_b32_e32 v9, 0x400000, v30
	v_cmp_u_f32_e32 vcc_lo, v30, v30
	v_add3_u32 v2, v2, v29, 0x7fff
	v_or_b32_e32 v30, 0x400000, v27
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v29, v29
	v_or_b32_e32 v29, 0x400000, v28
	v_bfe_u32 v9, v27, 16, 1
	v_cndmask_b32_e32 v10, v2, v10, vcc_lo
	v_add3_u32 v2, v31, v28, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v28, v28
	v_add3_u32 v9, v9, v27, 0x7fff
	v_bfe_u32 v31, v26, 16, 1
	v_cndmask_b32_e32 v28, v2, v29, vcc_lo
	v_add_nc_u32_e32 v2, s2, v3
	v_cmp_u_f32_e32 vcc_lo, v27, v27
	v_add3_u32 v27, v31, v26, 0x7fff
	v_or_b32_e32 v29, 0x400000, v26
	v_ashrrev_i32_e32 v3, 31, v2
	v_cndmask_b32_e32 v9, v9, v30, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v4, off
	global_store_d16_hi_b16 v[0:1], v5, off offset:4
	global_store_d16_hi_b16 v[0:1], v6, off offset:8
	global_store_d16_hi_b16 v[0:1], v7, off offset:12
	global_store_d16_hi_b16 v[0:1], v8, off offset:16
	global_store_d16_hi_b16 v[0:1], v10, off offset:20
	global_store_d16_hi_b16 v[0:1], v28, off offset:24
	global_store_d16_hi_b16 v[0:1], v9, off offset:28
	v_cmp_u_f32_e32 vcc_lo, v26, v26
	v_bfe_u32 v30, v25, 16, 1
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_or_b32_e32 v5, 0x400000, v25
	v_bfe_u32 v6, v24, 16, 1
	v_cndmask_b32_e32 v4, v27, v29, vcc_lo
	v_add3_u32 v3, v30, v25, 0x7fff
	v_bfe_u32 v7, v23, 16, 1
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v25, v25
	v_add3_u32 v7, v7, v23, 0x7fff
	v_or_b32_e32 v8, 0x400000, v23
	v_bfe_u32 v9, v22, 16, 1
	v_or_b32_e32 v10, 0x400000, v21
	v_cndmask_b32_e32 v5, v3, v5, vcc_lo
	v_add3_u32 v3, v6, v24, 0x7fff
	v_or_b32_e32 v6, 0x400000, v24
	v_cmp_u_f32_e32 vcc_lo, v24, v24
	v_add_nc_u32_e32 v2, s2, v2
	v_cndmask_b32_e32 v6, v3, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v23, v23
	v_bfe_u32 v3, v21, 16, 1
	v_bfe_u32 v23, v20, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v22, 0x7fff
	v_or_b32_e32 v9, 0x400000, v22
	v_cmp_u_f32_e32 vcc_lo, v22, v22
	v_add3_u32 v3, v3, v21, 0x7fff
	v_or_b32_e32 v22, 0x400000, v19
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v21, v21
	v_bfe_u32 v9, v19, 16, 1
	v_or_b32_e32 v21, 0x400000, v20
	v_cndmask_b32_e32 v10, v3, v10, vcc_lo
	v_add3_u32 v3, v23, v20, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v20, v20
	v_add3_u32 v9, v9, v19, 0x7fff
	v_bfe_u32 v23, v18, 16, 1
	v_cndmask_b32_e32 v20, v3, v21, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	v_ashrrev_i32_e32 v3, 31, v2
	v_add3_u32 v19, v23, v18, 0x7fff
	v_or_b32_e32 v21, 0x400000, v18
	v_cndmask_b32_e32 v9, v9, v22, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v4, off
	global_store_d16_hi_b16 v[0:1], v5, off offset:4
	global_store_d16_hi_b16 v[0:1], v6, off offset:8
	global_store_d16_hi_b16 v[0:1], v7, off offset:12
	global_store_d16_hi_b16 v[0:1], v8, off offset:16
	global_store_d16_hi_b16 v[0:1], v10, off offset:20
	global_store_d16_hi_b16 v[0:1], v20, off offset:24
	global_store_d16_hi_b16 v[0:1], v9, off offset:28
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_cmp_u_f32_e32 vcc_lo, v18, v18
	v_bfe_u32 v22, v17, 16, 1
	v_or_b32_e32 v4, 0x400000, v17
	v_bfe_u32 v5, v16, 16, 1
	v_bfe_u32 v6, v15, 16, 1
	v_cndmask_b32_e32 v2, v19, v21, vcc_lo
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add3_u32 v3, v22, v17, 0x7fff
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v17, v17
	v_bfe_u32 v7, v14, 16, 1
	v_add3_u32 v6, v6, v15, 0x7fff
	v_or_b32_e32 v8, 0x400000, v15
	v_bfe_u32 v9, v12, 16, 1
	v_cndmask_b32_e32 v3, v3, v4, vcc_lo
	v_add3_u32 v4, v5, v16, 0x7fff
	v_or_b32_e32 v5, 0x400000, v16
	v_cmp_u_f32_e32 vcc_lo, v16, v16
	v_or_b32_e32 v10, 0x400000, v13
	v_add3_u32 v9, v9, v12, 0x7fff
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v15, v15
	v_add3_u32 v5, v7, v14, 0x7fff
	v_or_b32_e32 v7, 0x400000, v14
	v_or_b32_e32 v15, 0x400000, v11
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo
	v_bfe_u32 v8, v13, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v14, v14
	v_or_b32_e32 v14, 0x400000, v12
	v_add3_u32 v8, v8, v13, 0x7fff
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_bfe_u32 v7, v11, 16, 1
	v_cndmask_b32_e32 v8, v8, v10, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v12, v12
	v_add3_u32 v7, v7, v11, 0x7fff
	v_cndmask_b32_e32 v9, v9, v14, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v11, v11
	v_cndmask_b32_e32 v7, v7, v15, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v2, off
	global_store_d16_hi_b16 v[0:1], v3, off offset:4
	global_store_d16_hi_b16 v[0:1], v4, off offset:8
	global_store_d16_hi_b16 v[0:1], v6, off offset:12
	global_store_d16_hi_b16 v[0:1], v5, off offset:16
	global_store_d16_hi_b16 v[0:1], v8, off offset:20
	global_store_d16_hi_b16 v[0:1], v9, off offset:24
	global_store_d16_hi_b16 v[0:1], v7, off offset:28
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
