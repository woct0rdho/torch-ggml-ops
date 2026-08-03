// Frozen instruction body from the project-owned HIP Q6_K exact-shape control.
// Source: build/mmq_bundle_sources/gfx1151/DenseFwdQ6KK2048J128Full-37056fa0ef1d238c.cu
// Compiler: AMD clang 23.0.0git, revision 998a696feca902c689c53383e48d29494f9edba7.
; %bb.0:
	v_and_b32_e32 v1, 0x3ff, v0
	v_dual_mov_b32 v14, 0 :: v_dual_and_b32 v3, 0x70, v0
	v_bfe_u32 v4, v0, 10, 10
	v_dual_mov_b32 v16, 0 :: v_dual_and_b32 v61, 15, v0
	v_bfe_u32 v5, v0, 2, 8
	v_dual_mov_b32 v24, 0 :: v_dual_and_b32 v9, 3, v0
	v_bfe_u32 v0, v0, 4, 6
	v_lshlrev_b32_e32 v68, 3, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v18, 0 :: v_dual_and_b32 v69, 2, v5
	v_dual_mov_b32 v17, 0 :: v_dual_lshlrev_b32 v10, 1, v9
	s_delay_alu instid0(VALU_DEP_4)
	v_lshl_or_b32 v0, v4, 4, v0
	v_mov_b32_e32 v22, 0
	v_lshl_add_u32 v9, v9, 2, 0
	s_clause 0x2
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x10
	s_load_b32 s22, s[0:1], 0x20
	v_mul_u32_u24_e32 v8, 0x130, v4
	v_mul_u32_u24_e32 v13, 0x4c, v0
	v_mov_b32_e32 v26, 0
	v_mad_u32_u24 v77, 0x90, v61, 0
	v_dual_mov_b32 v32, 0 :: v_dual_lshlrev_b32 v83, 1, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_lshl_add_u32 v78, v13, 2, 0
	v_mov_b32_e32 v13, 0
	v_dual_mov_b32 v28, 0 :: v_dual_add_nc_u32 v5, v5, v68
	v_dual_mov_b32 v47, 0 :: v_dual_add_nc_u32 v106, 0x4a00, v78
	v_dual_mov_b32 v50, 0 :: v_dual_add_nc_u32 v107, 0x4f00, v78
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v5, 63, v5
	v_dual_mov_b32 v49, 0 :: v_dual_add_nc_u32 v108, 0x5400, v78
	v_mov_b32_e32 v11, 0
	v_dual_mov_b32 v52, 0 :: v_dual_add_nc_u32 v109, 0x5800, v78
	v_xor_b32_e32 v12, 32, v5
	v_lshlrev_b32_e32 v73, 3, v5
	v_mul_u32_u24_e32 v5, 0x130, v5
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v74, 3, v12
	v_mul_u32_u24_e32 v12, 0x130, v12
	v_dual_mov_b32 v34, 0 :: v_dual_add_nc_u32 v85, v9, v5
	v_mov_b32_e32 v15, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v86, v9, v12
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v12, 0
	v_lshlrev_b32_e32 v2, 1, v1
	v_lshl_add_u32 v1, v4, 5, v1
	v_mad_u32_u24 v4, 0x1300, v4, 0
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v25, 0
	v_sub_nc_u32_e32 v6, v2, v61
	v_and_b32_e32 v7, 63, v1
	v_and_or_b32 v3, v2, 14, v3
	v_lshlrev_b32_e32 v80, 1, v2
	v_mad_u32_u24 v75, 0x130, v61, v4
	v_lshl_add_u32 v6, v6, 2, 0
	v_lshlrev_b32_e32 v70, 3, v7
	v_mul_u32_u24_e32 v7, 0x130, v7
	v_lshl_add_u32 v79, v1, 2, 0
	v_dual_mov_b32 v30, 0 :: v_dual_lshlrev_b32 v81, 1, v3
	v_add_nc_u32_e32 v2, v6, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v29, 0 :: v_dual_add_nc_u32 v82, 0, v7
	v_mov_b32_e32 v31, 0
	v_dual_mov_b32 v33, 0 :: v_dual_add_nc_u32 v92, 0x5c00, v2
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v87, 0x4800, v2
	v_add_nc_u32_e32 v88, 0x4c00, v2
	v_dual_mov_b32 v38, 0 :: v_dual_add_nc_u32 v89, 0x5000, v2
	v_add_nc_u32_e32 v90, 0x5800, v2
	v_dual_mov_b32 v40, 0 :: v_dual_add_nc_u32 v93, 0x6000, v2
	v_dual_mov_b32 v35, 0 :: v_dual_add_nc_u32 v94, 0x6400, v2
	v_dual_mov_b32 v42, 0 :: v_dual_add_nc_u32 v95, 0x6800, v2
	v_dual_mov_b32 v37, 0 :: v_dual_add_nc_u32 v96, 0x7000, v2
	v_dual_mov_b32 v39, 0 :: v_dual_add_nc_u32 v98, 0x7400, v2
	v_dual_mov_b32 v44, 0 :: v_dual_add_nc_u32 v99, 0x7800, v2
	v_dual_mov_b32 v41, 0 :: v_dual_add_nc_u32 v100, 0x7c00, v2
	v_dual_mov_b32 v46, 0 :: v_dual_add_nc_u32 v101, 0x8000, v2
	v_dual_mov_b32 v43, 0 :: v_dual_add_nc_u32 v102, 0x8600, v2
	v_dual_mov_b32 v48, 0 :: v_dual_add_nc_u32 v103, 0x8c00, v2
	v_dual_mov_b32 v45, 0 :: v_dual_add_nc_u32 v104, 0x9000, v2
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v97, 0 :: v_dual_mov_b32 v110, 0
	v_mov_b32_e32 v105, 0
	s_mov_b32 s4, 0
	s_lshl_b32 s3, s3, 7
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
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_i32 s18, s23, s20
	s_mul_i32 s19, s18, 0xd2
	s_mul_hi_i32 s24, s18, 0xd2
	s_add_u32 s18, s12, s19
	s_addc_u32 s19, s13, s24
	s_mul_i32 s24, s21, s23
	v_mad_u64_u32 v[4:5], null, 0xd2, v68, s[18:19]
	s_add_i32 s24, s24, s3
	v_mad_u64_u32 v[137:138], null, 0xd2, v74, s[18:19]
	v_mad_u64_u32 v[139:140], null, 0xd2, v70, s[18:19]
	v_add_co_u32 v2, vcc_lo, v4, v80
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, 0, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v81
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v6, vcc_lo, 0x1000, v2
	v_add_co_ci_u32_e64 v7, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v113, vcc_lo, 0x1000, v4
	v_add_co_ci_u32_e64 v114, null, 0, v5, vcc_lo
	v_add_co_u32 v115, vcc_lo, 0x3000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v116, null, 0, v3, vcc_lo
	v_add_co_u32 v117, vcc_lo, 0x3000, v4
	v_add_co_ci_u32_e64 v118, null, 0, v5, vcc_lo
	v_add_co_u32 v119, vcc_lo, 0x4000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v120, null, 0, v3, vcc_lo
	v_add_co_u32 v121, vcc_lo, 0x4000, v4
	v_add_co_ci_u32_e64 v122, null, 0, v5, vcc_lo
	s_clause 0x7
	global_load_b32 v112, v[2:3], off
	global_load_b32 v172, v[4:5], off offset:128
	global_load_b32 v9, v[6:7], off offset:2624
	global_load_b32 v10, v[113:114], off offset:2752
	global_load_b32 v7, v[115:116], off offset:1152
	global_load_b32 v8, v[117:118], off offset:1280
	global_load_b32 v6, v[119:120], off offset:3776
	global_load_b32 v111, v[121:122], off offset:3904
	v_add_co_u32 v113, vcc_lo, 0x6000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v114, null, 0, v3, vcc_lo
	v_add_co_u32 v115, vcc_lo, 0x6000, v4
	v_add_co_ci_u32_e64 v116, null, 0, v5, vcc_lo
	v_add_co_u32 v117, vcc_lo, 0x8000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v118, null, 0, v3, vcc_lo
	v_add_co_u32 v121, vcc_lo, 0x8000, v4
	v_add_co_ci_u32_e64 v122, null, 0, v5, vcc_lo
	v_add_co_u32 v123, vcc_lo, 0x9000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v124, null, 0, v3, vcc_lo
	v_add_co_u32 v125, vcc_lo, 0x9000, v4
	v_add_co_ci_u32_e64 v126, null, 0, v5, vcc_lo
	v_add_co_u32 v127, vcc_lo, 0xb000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v128, null, 0, v3, vcc_lo
	v_add_co_u32 v129, vcc_lo, 0xb000, v4
	v_add_co_ci_u32_e64 v130, null, 0, v5, vcc_lo
	s_clause 0x7
	global_load_b32 v119, v[113:114], off offset:2304
	global_load_b32 v120, v[115:116], off offset:2432
	global_load_b32 v116, v[117:118], off offset:832
	global_load_b32 v117, v[121:122], off offset:960
	global_load_b32 v114, v[123:124], off offset:3456
	global_load_b32 v115, v[125:126], off offset:3584
	global_load_b32 v113, v[127:128], off offset:1984
	global_load_b32 v118, v[129:130], off offset:2112
	v_add_co_u32 v121, vcc_lo, 0xd000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v122, null, 0, v3, vcc_lo
	v_add_co_u32 v123, vcc_lo, 0xd000, v4
	v_add_co_ci_u32_e64 v124, null, 0, v5, vcc_lo
	v_add_co_u32 v125, vcc_lo, 0xe000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v126, null, 0, v3, vcc_lo
	v_add_co_u32 v127, vcc_lo, 0xe000, v4
	v_add_co_ci_u32_e64 v128, null, 0, v5, vcc_lo
	v_add_co_u32 v129, vcc_lo, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v130, null, 0, v3, vcc_lo
	v_add_co_u32 v131, vcc_lo, 0x10000, v4
	v_add_co_ci_u32_e64 v132, null, 0, v5, vcc_lo
	v_add_co_u32 v133, vcc_lo, 0x12000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v134, null, 0, v3, vcc_lo
	v_add_co_u32 v135, vcc_lo, 0x12000, v4
	v_add_co_ci_u32_e64 v136, null, 0, v5, vcc_lo
	s_clause 0x7
	global_load_b32 v162, v[121:122], off offset:512
	global_load_b32 v163, v[123:124], off offset:640
	global_load_b32 v159, v[125:126], off offset:3136
	global_load_b32 v160, v[127:128], off offset:3264
	global_load_b32 v157, v[129:130], off offset:1664
	global_load_b32 v158, v[131:132], off offset:1792
	global_load_b32 v156, v[133:134], off offset:192
	global_load_b32 v161, v[135:136], off offset:320
	v_add_co_u32 v121, vcc_lo, 0x13000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v122, null, 0, v3, vcc_lo
	v_add_co_u32 v123, vcc_lo, 0x13000, v4
	v_add_co_ci_u32_e64 v124, null, 0, v5, vcc_lo
	v_add_co_u32 v125, vcc_lo, 0x15000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v126, null, 0, v3, vcc_lo
	v_add_co_u32 v127, vcc_lo, 0x15000, v4
	v_add_co_ci_u32_e64 v128, null, 0, v5, vcc_lo
	v_add_co_u32 v129, vcc_lo, 0x16000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v130, null, 0, v3, vcc_lo
	v_add_co_u32 v131, vcc_lo, 0x17000, v4
	v_add_co_ci_u32_e64 v132, null, 0, v5, vcc_lo
	v_add_co_u32 v133, vcc_lo, 0x18000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v134, null, 0, v3, vcc_lo
	v_mad_u64_u32 v[2:3], null, s24, 36, v[1:2]
	v_mad_u64_u32 v[135:136], null, 0xd2, v73, s[18:19]
	v_add_co_u32 v4, vcc_lo, 0x18000, v4
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_mov_b32 s18, -4
	v_ashrrev_i32_e32 v3, 31, v2
	v_add_co_u32 v135, vcc_lo, v135, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v136, null, 0, v136, vcc_lo
	v_lshlrev_b64 v[141:142], 2, v[2:3]
	v_add_co_u32 v137, vcc_lo, v137, v83
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v138, null, 0, v138, vcc_lo
	v_add_co_u32 v141, vcc_lo, s14, v141
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v142, null, s15, v142, vcc_lo
	s_clause 0xa
	global_load_b32 v170, v[121:122], off offset:2816
	global_load_b32 v171, v[123:124], off offset:2944
	global_load_b32 v168, v[125:126], off offset:1344
	global_load_b32 v169, v[127:128], off offset:1472
	global_load_b32 v166, v[129:130], off offset:3968
	global_load_b32 v167, v[131:132], off
	global_load_b32 v164, v[133:134], off offset:2496
	global_load_b32 v165, v[4:5], off offset:2624
	global_load_d16_b16 v174, v[139:140], off offset:208
	global_load_b32 v3, v[135:136], off offset:192
	global_load_b32 v4, v[137:138], off offset:192
	s_clause 0x7
	global_load_b32 v5, v[141:142], off
	global_load_b32 v121, v[141:142], off offset:512
	global_load_b32 v122, v[141:142], off offset:1024
	global_load_b32 v123, v[141:142], off offset:1536
	global_load_b32 v124, v[141:142], off offset:2048
	global_load_b32 v125, v[141:142], off offset:2560
	global_load_b32 v126, v[141:142], off offset:3072
	global_load_b32 v127, v[141:142], off offset:3584
	v_add_co_u32 v134, vcc_lo, 0x1000, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v135, null, 0, v142, vcc_lo
	v_add_co_u32 v144, vcc_lo, v141, 0x2000
	v_add_co_ci_u32_e64 v145, null, 0, v142, vcc_lo
	v_add_co_u32 v146, vcc_lo, 0x2000, v141
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v147, null, 0, v142, vcc_lo
	v_add_co_u32 v151, vcc_lo, 0x3000, v141
	v_add_co_ci_u32_e64 v152, null, 0, v142, vcc_lo
	v_add_co_u32 v148, vcc_lo, v141, 0x4000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v149, null, 0, v142, vcc_lo
	v_add_co_u32 v175, vcc_lo, 0x4000, v141
	v_add_co_ci_u32_e64 v176, null, 0, v142, vcc_lo
	s_clause 0x1b
	global_load_b32 v128, v[134:135], off offset:512
	global_load_b32 v129, v[134:135], off offset:1024
	global_load_b32 v130, v[134:135], off offset:1536
	global_load_b32 v131, v[134:135], off offset:2048
	global_load_b32 v132, v[134:135], off offset:2560
	global_load_b32 v133, v[134:135], off offset:3072
	global_load_b32 v134, v[134:135], off offset:3584
	global_load_b32 v135, v[146:147], off offset:512
	global_load_b32 v136, v[146:147], off offset:1024
	global_load_b32 v137, v[146:147], off offset:1536
	global_load_b32 v138, v[146:147], off offset:2048
	global_load_b32 v139, v[146:147], off offset:2560
	global_load_b32 v140, v[146:147], off offset:3072
	global_load_b32 v141, v[146:147], off offset:3584
	global_load_b32 v142, v[151:152], off offset:512
	global_load_b32 v143, v[151:152], off offset:1024
	global_load_b32 v153, v[144:145], off offset:-4096
	global_load_b32 v146, v[144:145], off
	global_load_b32 v145, v[148:149], off offset:-4096
	global_load_b32 v144, v[148:149], off
	global_load_b32 v147, v[151:152], off offset:1536
	global_load_b32 v148, v[151:152], off offset:2048
	global_load_b32 v149, v[151:152], off offset:2560
	global_load_b32 v150, v[151:152], off offset:3072
	global_load_b32 v151, v[151:152], off offset:3584
	global_load_b32 v152, v[175:176], off offset:512
	global_load_b32 v154, v[175:176], off offset:1024
	global_load_b32 v155, v[175:176], off offset:1536
	s_waitcnt vmcnt(62)
	v_lshrrev_b32_e32 v173, 4, v112
	v_ashrrev_i32_e32 v172, v69, v172
	v_and_b32_e32 v112, 0xf0f0f0f, v112
	v_and_b32_e32 v175, 0xf0f0f0f, v9
	v_lshrrev_b32_e32 v9, 4, v9
	v_and_b32_e32 v173, 0xf0f0f0f, v173
	v_ashrrev_i32_e32 v10, v69, v10
	v_and_b32_e32 v176, 0xf0f0f0f, v7
	v_lshrrev_b32_e32 v7, 4, v7
	v_ashrrev_i32_e32 v8, v69, v8
	v_and_b32_e32 v177, 0xf0f0f0f, v6
	v_lshrrev_b32_e32 v6, 4, v6
	v_ashrrev_i32_e32 v111, v69, v111
	v_and_b32_e32 v9, 0xf0f0f0f, v9
	v_and_b32_e32 v7, 0xf0f0f0f, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_and_b32_e32 v6, 0xf0f0f0f, v6
	v_lshlrev_b32_e32 v190, 4, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v177, 0x30303030, v190, v177
	v_and_or_b32 v190, 0x30303030, v111, v6
	v_and_b32_e32 v178, 0xf0f0f0f, v119
	v_lshrrev_b32_e32 v119, 4, v119
	s_waitcnt vmcnt(61)
	v_ashrrev_i32_e32 v120, v69, v120
	s_waitcnt vmcnt(60)
	v_and_b32_e32 v179, 0xf0f0f0f, v116
	v_lshrrev_b32_e32 v116, 4, v116
	s_waitcnt vmcnt(59)
	v_ashrrev_i32_e32 v117, v69, v117
	s_waitcnt vmcnt(58)
	v_and_b32_e32 v180, 0xf0f0f0f, v114
	v_lshrrev_b32_e32 v114, 4, v114
	s_waitcnt vmcnt(57)
	v_ashrrev_i32_e32 v115, v69, v115
	s_waitcnt vmcnt(56)
	v_and_b32_e32 v181, 0xf0f0f0f, v113
	v_lshrrev_b32_e32 v113, 4, v113
	s_waitcnt vmcnt(55)
	v_ashrrev_i32_e32 v118, v69, v118
	v_lshlrev_b32_e32 v191, 4, v120
	v_and_b32_e32 v119, 0xf0f0f0f, v119
	v_lshlrev_b32_e32 v192, 4, v117
	v_and_b32_e32 v116, 0xf0f0f0f, v116
	v_lshlrev_b32_e32 v193, 4, v115
	v_and_b32_e32 v114, 0xf0f0f0f, v114
	v_lshlrev_b32_e32 v194, 4, v118
	v_and_b32_e32 v113, 0xf0f0f0f, v113
	v_and_or_b32 v178, 0x30303030, v191, v178
	v_and_or_b32 v191, 0x30303030, v120, v119
	v_and_or_b32 v179, 0x30303030, v192, v179
	v_and_or_b32 v192, 0x30303030, v117, v116
	v_and_or_b32 v180, 0x30303030, v193, v180
	v_and_or_b32 v193, 0x30303030, v115, v114
	v_and_or_b32 v181, 0x30303030, v194, v181
	v_and_or_b32 v194, 0x30303030, v118, v113
	v_lshlrev_b16 v118.l, 8, v177.l
	v_lshlrev_b16 v118.h, 8, v177.h
	v_lshlrev_b16 v119.h, 8, v190.l
	v_lshlrev_b16 v120.h, 8, v190.h
	v_and_b16 v119.l, 0x3f00, v177.l
	s_waitcnt vmcnt(54)
	v_and_b32_e32 v182, 0xf0f0f0f, v162
	v_lshrrev_b32_e32 v162, 4, v162
	s_waitcnt vmcnt(53)
	v_ashrrev_i32_e32 v163, v69, v163
	s_waitcnt vmcnt(52)
	v_and_b32_e32 v183, 0xf0f0f0f, v159
	v_lshrrev_b32_e32 v159, 4, v159
	s_waitcnt vmcnt(51)
	v_ashrrev_i32_e32 v160, v69, v160
	s_waitcnt vmcnt(50)
	v_and_b32_e32 v184, 0xf0f0f0f, v157
	v_lshrrev_b32_e32 v157, 4, v157
	s_waitcnt vmcnt(49)
	v_ashrrev_i32_e32 v158, v69, v158
	s_waitcnt vmcnt(48)
	v_and_b32_e32 v185, 0xf0f0f0f, v156
	v_lshrrev_b32_e32 v156, 4, v156
	s_waitcnt vmcnt(47)
	v_ashrrev_i32_e32 v161, v69, v161
	v_lshlrev_b32_e32 v195, 4, v163
	v_and_b32_e32 v162, 0xf0f0f0f, v162
	v_lshlrev_b32_e32 v196, 4, v160
	v_and_b32_e32 v159, 0xf0f0f0f, v159
	v_lshlrev_b32_e32 v197, 4, v158
	v_and_b32_e32 v157, 0xf0f0f0f, v157
	v_lshlrev_b32_e32 v198, 4, v161
	v_and_b32_e32 v156, 0xf0f0f0f, v156
	v_and_or_b32 v182, 0x30303030, v195, v182
	v_and_or_b32 v195, 0x30303030, v163, v162
	v_and_or_b32 v183, 0x30303030, v196, v183
	v_and_or_b32 v196, 0x30303030, v160, v159
	v_and_or_b32 v184, 0x30303030, v197, v184
	v_and_or_b32 v197, 0x30303030, v158, v157
	v_and_or_b32 v198, 0x30303030, v198, v185
	v_and_or_b32 v203, 0x30303030, v161, v156
	v_lshlrev_b16 v156.h, 8, v178.l
	v_lshlrev_b16 v157.h, 8, v178.h
	v_lshlrev_b16 v158.l, 8, v191.l
	v_lshlrev_b16 v159.h, 8, v191.h
	v_lshlrev_b16 v160.l, 8, v179.l
	v_lshlrev_b16 v161.h, 8, v179.h
	v_lshlrev_b16 v162.l, 8, v192.l
	v_lshlrev_b16 v163.l, 8, v192.h
	v_and_b16 v120.l, 0x3f00, v177.h
	v_and_b16 v156.l, 0x3f00, v190.l
	v_and_b16 v157.l, 0x3f00, v190.h
	v_and_b16 v158.h, 0x3f00, v178.l
	v_and_b16 v159.l, 0x3f00, v178.h
	s_waitcnt vmcnt(46)
	v_and_b32_e32 v186, 0xf0f0f0f, v170
	v_lshrrev_b32_e32 v170, 4, v170
	s_waitcnt vmcnt(45)
	v_ashrrev_i32_e32 v171, v69, v171
	s_waitcnt vmcnt(44)
	v_and_b32_e32 v187, 0xf0f0f0f, v168
	v_lshrrev_b32_e32 v168, 4, v168
	s_waitcnt vmcnt(43)
	v_ashrrev_i32_e32 v169, v69, v169
	s_waitcnt vmcnt(42)
	v_and_b32_e32 v188, 0xf0f0f0f, v166
	v_lshrrev_b32_e32 v166, 4, v166
	s_waitcnt vmcnt(38)
	v_cvt_f32_f16_e64 v205, v174.l
	v_lshlrev_b32_e32 v174, 4, v172
	v_ashrrev_i32_e32 v167, v69, v167
	v_and_b32_e32 v189, 0xf0f0f0f, v164
	v_lshrrev_b32_e32 v164, 4, v164
	v_ashrrev_i32_e32 v165, v69, v165
	v_and_or_b32 v112, 0x30303030, v174, v112
	v_and_or_b32 v172, 0x30303030, v172, v173
	v_lshlrev_b32_e32 v173, 4, v10
	v_lshlrev_b32_e32 v174, 4, v8
	v_and_b32_e32 v170, 0xf0f0f0f, v170
	v_lshlrev_b32_e32 v199, 4, v171
	v_and_b32_e32 v168, 0xf0f0f0f, v168
	v_lshlrev_b32_e32 v200, 4, v169
	v_and_b32_e32 v166, 0xf0f0f0f, v166
	v_lshlrev_b32_e32 v201, 4, v167
	v_and_b32_e32 v164, 0xf0f0f0f, v164
	v_lshlrev_b32_e32 v202, 4, v165
	v_and_or_b32 v173, 0x30303030, v173, v175
	v_and_or_b32 v175, 0x30303030, v10, v9
	v_and_or_b32 v174, 0x30303030, v174, v176
	v_and_or_b32 v176, 0x30303030, v8, v7
	v_lshlrev_b16 v6.h, 8, v112.l
	v_lshlrev_b16 v7.h, 8, v112.h
	v_lshlrev_b16 v8.h, 8, v172.l
	v_lshlrev_b16 v9.l, 8, v172.h
	v_and_or_b32 v199, 0x30303030, v199, v186
	v_and_or_b32 v204, 0x30303030, v171, v170
	v_and_or_b32 v200, 0x30303030, v200, v187
	v_and_or_b32 v206, 0x30303030, v169, v168
	v_and_or_b32 v201, 0x30303030, v201, v188
	v_and_b16 v7.l, 0x3f00, v112.h
	v_and_or_b32 v207, 0x30303030, v167, v166
	v_and_or_b32 v208, 0x30303030, v202, v189
	v_and_or_b32 v209, 0x30303030, v165, v164
	v_add_nc_u16 v6.h, 0xe000, v6.h
	v_add_nc_u16 v7.h, 0xe000, v7.h
	v_add_nc_u16 v8.h, 0xe000, v8.h
	v_add_nc_u16 v9.l, 0xe000, v9.l
	v_lshlrev_b16 v10.h, 8, v173.l
	v_lshlrev_b16 v111.h, 8, v173.h
	v_lshlrev_b16 v112.h, 8, v175.l
	v_lshlrev_b16 v113.h, 8, v175.h
	v_lshlrev_b16 v114.h, 8, v174.l
	v_lshlrev_b16 v115.h, 8, v174.h
	v_lshlrev_b16 v116.h, 8, v176.l
	v_lshlrev_b16 v117.l, 8, v176.h
	v_and_b16 v6.l, 0x3f00, v112.l
	v_and_b16 v8.l, 0x3f00, v172.l
	v_and_b16 v9.h, 0x3f00, v172.h
	v_and_b16 v10.l, 0x3f00, v173.l
	v_and_b16 v111.l, 0x3f00, v173.h
	v_and_b16 v112.l, 0x3f00, v175.l
	v_and_b16 v113.l, 0x3f00, v175.h
	v_and_b16 v114.l, 0x3f00, v174.l
	v_and_b16 v115.l, 0x3f00, v174.h
	v_and_b16 v116.l, 0x3f00, v176.l
	v_and_b16 v117.h, 0x3f00, v176.h
	v_and_b16 v161.l, 0x3f00, v191.h
	v_and_b16 v163.h, 0x3f00, v179.h
	v_lshlrev_b16 v164.l, 8, v180.l
	v_and_b16 v164.h, 0x3f00, v192.l
	v_lshlrev_b16 v165.h, 8, v180.h
	v_lshlrev_b16 v166.l, 8, v193.l
	v_and_b16 v166.h, 0x3f00, v180.l
	v_lshlrev_b16 v167.h, 8, v193.h
	v_lshlrev_b16 v168.l, 8, v181.l
	v_and_b16 v168.h, 0x3f00, v193.l
	v_lshlrev_b16 v169.h, 8, v181.h
	v_lshlrev_b16 v170.l, 8, v194.l
	v_and_b16 v170.h, 0x3f00, v181.l
	v_and_b16 v171.l, 0x3f00, v181.h
	v_lshlrev_b16 v171.h, 8, v194.h
	v_lshlrev_b16 v172.l, 8, v182.l
	v_and_b16 v172.h, 0x3f00, v194.l
	v_lshlrev_b16 v173.l, 8, v182.h
	v_and_b16 v173.h, 0x3f00, v194.h
	v_lshlrev_b16 v174.l, 8, v195.l
	v_and_b16 v174.h, 0x3f00, v182.l
	v_and_b16 v175.l, 0x3f00, v182.h
	v_lshlrev_b16 v175.h, 8, v195.h
	v_lshlrev_b16 v176.l, 8, v183.l
	v_and_b16 v176.h, 0x3f00, v195.l
	v_and_b16 v177.l, 0x3f00, v195.h
	v_lshlrev_b16 v177.h, 8, v183.h
	v_lshlrev_b16 v178.l, 8, v196.l
	v_and_b16 v178.h, 0x3f00, v183.l
	v_lshlrev_b16 v179.h, 8, v196.h
	v_lshlrev_b16 v180.l, 8, v184.l
	v_and_b16 v181.l, 0x3f00, v196.h
	v_lshlrev_b16 v181.h, 8, v184.h
	v_lshlrev_b16 v182.l, 8, v197.l
	v_and_b16 v182.h, 0x3f00, v184.l
	v_lshlrev_b16 v183.l, 8, v197.h
	v_lshlrev_b16 v184.l, 8, v198.l
	v_and_b16 v185.l, 0x3f00, v197.h
	v_lshlrev_b16 v185.h, 8, v198.h
	v_lshlrev_b16 v186.l, 8, v203.l
	v_and_b16 v186.h, 0x3f00, v198.l
	v_lshlrev_b16 v187.h, 8, v203.h
	v_lshlrev_b16 v188.l, 8, v199.l
	v_lshlrev_b16 v189.h, 8, v199.h
	v_lshlrev_b16 v190.l, 8, v204.l
	v_and_b16 v190.h, 0x3f00, v199.l
	v_lshlrev_b16 v191.h, 8, v204.h
	v_lshlrev_b16 v192.l, 8, v200.l
	v_lshlrev_b16 v193.l, 8, v200.h
	v_lshlrev_b16 v194.l, 8, v206.l
	v_and_b16 v194.h, 0x3f00, v200.l
	v_and_b16 v195.l, 0x3f00, v200.h
	v_lshlrev_b16 v195.h, 8, v206.h
	v_lshlrev_b16 v196.h, 8, v201.l
	v_lshlrev_b16 v197.h, 8, v201.h
	v_lshlrev_b16 v198.l, 8, v207.l
	v_and_b16 v199.l, 0x3f00, v201.h
	v_lshlrev_b16 v200.l, 8, v207.h
	v_lshlrev_b16 v200.h, 8, v208.l
	v_lshlrev_b16 v201.h, 8, v208.h
	v_lshlrev_b16 v202.l, 8, v209.l
	v_lshlrev_b16 v202.h, 8, v209.h
	v_lshrrev_b16 v6.h, 8, v6.h
	v_lshrrev_b16 v7.h, 8, v7.h
	v_lshrrev_b16 v8.h, 8, v8.h
	v_lshrrev_b16 v9.l, 8, v9.l
	v_add_nc_u16 v10.h, 0xe000, v10.h
	v_add_nc_u16 v111.h, 0xe000, v111.h
	v_add_nc_u16 v112.h, 0xe000, v112.h
	v_add_nc_u16 v113.h, 0xe000, v113.h
	v_add_nc_u16 v114.h, 0xe000, v114.h
	v_add_nc_u16 v115.h, 0xe000, v115.h
	v_add_nc_u16 v116.h, 0xe000, v116.h
	v_add_nc_u16 v117.l, 0xe000, v117.l
	v_add_nc_u16 v118.l, 0xe000, v118.l
	v_add_nc_u16 v118.h, 0xe000, v118.h
	v_add_nc_u16 v119.h, 0xe000, v119.h
	v_add_nc_u16 v120.h, 0xe000, v120.h
	v_add_nc_u16 v156.h, 0xe000, v156.h
	v_add_nc_u16 v157.h, 0xe000, v157.h
	v_add_nc_u16 v158.l, 0xe000, v158.l
	v_add_nc_u16 v159.h, 0xe000, v159.h
	v_add_nc_u16 v160.l, 0xe000, v160.l
	v_add_nc_u16 v161.h, 0xe000, v161.h
	v_add_nc_u16 v162.l, 0xe000, v162.l
	v_add_nc_u16 v163.l, 0xe000, v163.l
	v_add_nc_u16 v164.l, 0xe000, v164.l
	v_add_nc_u16 v165.h, 0xe000, v165.h
	v_add_nc_u16 v166.l, 0xe000, v166.l
	v_add_nc_u16 v167.h, 0xe000, v167.h
	v_add_nc_u16 v168.l, 0xe000, v168.l
	v_add_nc_u16 v169.h, 0xe000, v169.h
	v_add_nc_u16 v170.l, 0xe000, v170.l
	v_add_nc_u16 v171.h, 0xe000, v171.h
	v_add_nc_u16 v172.l, 0xe000, v172.l
	v_add_nc_u16 v173.l, 0xe000, v173.l
	v_add_nc_u16 v174.l, 0xe000, v174.l
	v_add_nc_u16 v175.h, 0xe000, v175.h
	v_add_nc_u16 v176.l, 0xe000, v176.l
	v_add_nc_u16 v177.h, 0xe000, v177.h
	v_add_nc_u16 v178.l, 0xe000, v178.l
	v_add_nc_u16 v179.h, 0xe000, v179.h
	v_add_nc_u16 v180.l, 0xe000, v180.l
	v_add_nc_u16 v181.h, 0xe000, v181.h
	v_add_nc_u16 v182.l, 0xe000, v182.l
	v_add_nc_u16 v183.l, 0xe000, v183.l
	v_add_nc_u16 v184.l, 0xe000, v184.l
	v_add_nc_u16 v185.h, 0xe000, v185.h
	v_add_nc_u16 v186.l, 0xe000, v186.l
	v_add_nc_u16 v187.h, 0xe000, v187.h
	v_add_nc_u16 v188.l, 0xe000, v188.l
	v_add_nc_u16 v189.h, 0xe000, v189.h
	v_add_nc_u16 v190.l, 0xe000, v190.l
	v_add_nc_u16 v191.h, 0xe000, v191.h
	v_add_nc_u16 v192.l, 0xe000, v192.l
	v_add_nc_u16 v193.l, 0xe000, v193.l
	v_add_nc_u16 v194.l, 0xe000, v194.l
	v_add_nc_u16 v195.h, 0xe000, v195.h
	v_add_nc_u16 v196.h, 0xe000, v196.h
	v_add_nc_u16 v197.h, 0xe000, v197.h
	v_add_nc_u16 v198.l, 0xe000, v198.l
	v_add_nc_u16 v200.l, 0xe000, v200.l
	v_add_nc_u16 v200.h, 0xe000, v200.h
	v_add_nc_u16 v201.h, 0xe000, v201.h
	v_add_nc_u16 v202.l, 0xe000, v202.l
	v_add_nc_u16 v202.h, 0xe000, v202.h
	v_or_b16 v6.l, v6.l, v6.h
	v_or_b16 v6.h, v7.l, v7.h
	v_or_b16 v7.l, v8.l, v8.h
	v_or_b16 v7.h, v9.h, v9.l
	v_lshrrev_b16 v8.l, 8, v10.h
	v_lshrrev_b16 v8.h, 8, v111.h
	v_lshrrev_b16 v9.l, 8, v112.h
	v_lshrrev_b16 v9.h, 8, v113.h
	v_lshrrev_b16 v10.h, 8, v114.h
	v_lshrrev_b16 v111.h, 8, v115.h
	v_lshrrev_b16 v112.h, 8, v116.h
	v_lshrrev_b16 v113.h, 8, v117.l
	v_lshrrev_b16 v114.h, 8, v118.l
	v_lshrrev_b16 v115.h, 8, v118.h
	v_lshrrev_b16 v116.h, 8, v119.h
	v_lshrrev_b16 v117.l, 8, v120.h
	v_and_b16 v160.h, 0x3f00, v191.l
	v_lshrrev_b16 v118.l, 8, v156.h
	v_lshrrev_b16 v118.h, 8, v157.h
	v_lshrrev_b16 v119.h, 8, v158.l
	v_lshrrev_b16 v120.h, 8, v159.h
	v_and_b16 v162.h, 0x3f00, v179.l
	v_and_b16 v165.l, 0x3f00, v192.h
	v_lshrrev_b16 v156.h, 8, v160.l
	v_lshrrev_b16 v157.h, 8, v161.h
	v_lshrrev_b16 v158.l, 8, v162.l
	v_lshrrev_b16 v159.h, 8, v163.l
	v_and_b16 v167.l, 0x3f00, v180.h
	v_and_b16 v169.l, 0x3f00, v193.h
	v_and_b16 v179.l, 0x3f00, v183.h
	v_and_b16 v180.h, 0x3f00, v196.l
	v_and_b16 v183.h, 0x3f00, v184.h
	v_and_b16 v184.h, 0x3f00, v197.l
	v_and_b16 v187.l, 0x3f00, v198.h
	v_and_b16 v188.h, 0x3f00, v203.l
	v_and_b16 v189.l, 0x3f00, v203.h
	v_and_b16 v191.l, 0x3f00, v199.h
	v_and_b16 v192.h, 0x3f00, v204.l
	v_and_b16 v193.h, 0x3f00, v204.h
	v_and_b16 v196.l, 0x3f00, v206.l
	v_and_b16 v197.l, 0x3f00, v206.h
	v_and_b16 v198.h, 0x3f00, v201.l
	v_and_b16 v199.h, 0x3f00, v207.l
	v_and_b16 v201.l, 0x3f00, v207.h
	v_and_b16 v203.l, 0x3f00, v208.l
	v_and_b16 v203.h, 0x3f00, v208.h
	v_and_b16 v204.l, 0x3f00, v209.l
	v_and_b16 v204.h, 0x3f00, v209.h
	v_lshrrev_b16 v160.l, 8, v164.l
	v_lshrrev_b16 v161.h, 8, v165.h
	v_lshrrev_b16 v162.l, 8, v166.l
	v_lshrrev_b16 v163.l, 8, v167.h
	v_lshrrev_b16 v164.l, 8, v168.l
	v_lshrrev_b16 v165.h, 8, v169.h
	v_lshrrev_b16 v166.l, 8, v170.l
	v_lshrrev_b16 v167.h, 8, v171.h
	v_lshrrev_b16 v168.l, 8, v172.l
	v_lshrrev_b16 v169.h, 8, v173.l
	v_lshrrev_b16 v170.l, 8, v174.l
	v_lshrrev_b16 v171.h, 8, v175.h
	v_lshrrev_b16 v172.l, 8, v176.l
	v_lshrrev_b16 v173.l, 8, v177.h
	v_lshrrev_b16 v174.l, 8, v178.l
	v_lshrrev_b16 v175.h, 8, v179.h
	v_lshrrev_b16 v176.l, 8, v180.l
	v_lshrrev_b16 v177.h, 8, v181.h
	v_lshrrev_b16 v178.l, 8, v182.l
	v_lshrrev_b16 v179.h, 8, v183.l
	v_lshrrev_b16 v180.l, 8, v184.l
	v_lshrrev_b16 v181.h, 8, v185.h
	v_lshrrev_b16 v182.l, 8, v186.l
	v_lshrrev_b16 v183.l, 8, v187.h
	v_lshrrev_b16 v184.l, 8, v188.l
	v_lshrrev_b16 v185.h, 8, v189.h
	v_lshrrev_b16 v186.l, 8, v190.l
	v_lshrrev_b16 v187.h, 8, v191.h
	v_lshrrev_b16 v188.l, 8, v192.l
	v_lshrrev_b16 v189.h, 8, v193.l
	v_lshrrev_b16 v190.l, 8, v194.l
	v_lshrrev_b16 v191.h, 8, v195.h
	v_lshrrev_b16 v192.l, 8, v196.h
	v_lshrrev_b16 v193.l, 8, v197.h
	v_lshrrev_b16 v194.l, 8, v198.l
	v_lshrrev_b16 v195.h, 8, v200.l
	v_lshrrev_b16 v196.h, 8, v200.h
	v_lshrrev_b16 v197.h, 8, v201.h
	v_lshrrev_b16 v198.l, 8, v202.l
	v_lshrrev_b16 v200.l, 8, v202.h
	v_add_nc_u16 v202.l, 0xe000, v6.l
	v_add_nc_u16 v202.h, 0xe000, v6.h
	v_add_nc_u16 v206.l, 0xe000, v7.l
	v_add_nc_u16 v206.h, 0xe000, v7.h
	v_or_b16 v6.l, v10.l, v8.l
	v_or_b16 v6.h, v111.l, v8.h
	v_or_b16 v7.l, v112.l, v9.l
	v_or_b16 v7.h, v113.l, v9.h
	v_or_b16 v8.l, v114.l, v10.h
	v_or_b16 v8.h, v115.l, v111.h
	v_or_b16 v9.l, v116.l, v112.h
	v_or_b16 v9.h, v117.h, v113.h
	v_or_b16 v10.l, v119.l, v114.h
	v_or_b16 v10.h, v120.l, v115.h
	v_or_b16 v111.l, v156.l, v116.h
	v_or_b16 v111.h, v157.l, v117.l
	v_or_b16 v112.l, v158.h, v118.l
	v_or_b16 v112.h, v159.l, v118.h
	v_or_b16 v113.l, v160.h, v119.h
	v_or_b16 v113.h, v161.l, v120.h
	v_or_b16 v114.l, v162.h, v156.h
	v_or_b16 v114.h, v163.h, v157.h
	v_or_b16 v115.l, v164.h, v158.l
	v_or_b16 v115.h, v165.l, v159.h
	v_or_b16 v116.l, v166.h, v160.l
	v_or_b16 v116.h, v167.l, v161.h
	v_or_b16 v117.l, v168.h, v162.l
	v_or_b16 v117.h, v169.l, v163.l
	v_or_b16 v118.l, v170.h, v164.l
	v_or_b16 v118.h, v171.l, v165.h
	v_or_b16 v119.l, v172.h, v166.l
	v_or_b16 v119.h, v173.h, v167.h
	v_or_b16 v120.l, v174.h, v168.l
	v_or_b16 v120.h, v175.l, v169.h
	v_or_b16 v156.l, v176.h, v170.l
	v_or_b16 v156.h, v177.l, v171.h
	v_or_b16 v157.l, v178.h, v172.l
	v_or_b16 v157.h, v179.l, v173.l
	v_or_b16 v158.l, v180.h, v174.l
	v_or_b16 v158.h, v181.l, v175.h
	v_or_b16 v159.l, v182.h, v176.l
	v_or_b16 v159.h, v183.h, v177.h
	v_or_b16 v160.l, v184.h, v178.l
	v_or_b16 v160.h, v185.l, v179.h
	v_or_b16 v161.l, v186.h, v180.l
	v_or_b16 v161.h, v187.l, v181.h
	v_or_b16 v162.l, v188.h, v182.l
	v_or_b16 v162.h, v189.l, v183.l
	v_or_b16 v163.l, v190.h, v184.l
	v_or_b16 v163.h, v191.l, v185.h
	v_or_b16 v164.l, v192.h, v186.l
	v_or_b16 v164.h, v193.h, v187.h
	v_or_b16 v165.l, v194.h, v188.l
	v_or_b16 v165.h, v195.l, v189.h
	v_or_b16 v166.l, v196.l, v190.l
	v_or_b16 v166.h, v197.l, v191.h
	v_or_b16 v167.l, v198.h, v192.l
	v_or_b16 v167.h, v199.l, v193.l
	v_or_b16 v168.l, v199.h, v194.l
	v_or_b16 v168.h, v201.l, v195.h
	v_or_b16 v169.l, v203.l, v196.h
	v_or_b16 v169.h, v203.h, v197.h
	v_or_b16 v170.l, v204.l, v198.l
	v_or_b16 v170.h, v204.h, v200.l
	v_add_nc_u16 v6.l, 0xe000, v6.l
	v_add_nc_u16 v6.h, 0xe000, v6.h
	v_add_nc_u16 v7.l, 0xe000, v7.l
	v_add_nc_u16 v7.h, 0xe000, v7.h
	v_add_nc_u16 v8.l, 0xe000, v8.l
	v_add_nc_u16 v8.h, 0xe000, v8.h
	v_add_nc_u16 v9.l, 0xe000, v9.l
	v_add_nc_u16 v9.h, 0xe000, v9.h
	v_add_nc_u16 v10.l, 0xe000, v10.l
	v_add_nc_u16 v10.h, 0xe000, v10.h
	v_add_nc_u16 v111.l, 0xe000, v111.l
	v_add_nc_u16 v111.h, 0xe000, v111.h
	v_add_nc_u16 v112.l, 0xe000, v112.l
	v_add_nc_u16 v112.h, 0xe000, v112.h
	v_add_nc_u16 v113.l, 0xe000, v113.l
	v_add_nc_u16 v113.h, 0xe000, v113.h
	v_add_nc_u16 v114.l, 0xe000, v114.l
	v_add_nc_u16 v114.h, 0xe000, v114.h
	v_add_nc_u16 v115.l, 0xe000, v115.l
	v_add_nc_u16 v115.h, 0xe000, v115.h
	ds_store_2addr_b32 v87, v202, v206 offset0:128 offset1:144
	v_add_nc_u16 v116.l, 0xe000, v116.l
	v_add_nc_u16 v116.h, 0xe000, v116.h
	v_add_nc_u16 v117.l, 0xe000, v117.l
	v_add_nc_u16 v117.h, 0xe000, v117.h
	v_add_nc_u16 v118.l, 0xe000, v118.l
	v_add_nc_u16 v118.h, 0xe000, v118.h
	v_add_nc_u16 v119.l, 0xe000, v119.l
	v_add_nc_u16 v119.h, 0xe000, v119.h
	v_add_nc_u16 v120.l, 0xe000, v120.l
	v_add_nc_u16 v120.h, 0xe000, v120.h
	v_add_nc_u16 v156.l, 0xe000, v156.l
	v_add_nc_u16 v156.h, 0xe000, v156.h
	v_add_nc_u16 v157.l, 0xe000, v157.l
	v_add_nc_u16 v157.h, 0xe000, v157.h
	v_add_nc_u16 v158.l, 0xe000, v158.l
	v_add_nc_u16 v158.h, 0xe000, v158.h
	v_add_nc_u16 v159.l, 0xe000, v159.l
	v_add_nc_u16 v159.h, 0xe000, v159.h
	v_add_nc_u16 v160.l, 0xe000, v160.l
	v_add_nc_u16 v160.h, 0xe000, v160.h
	v_add_nc_u16 v161.l, 0xe000, v161.l
	v_add_nc_u16 v161.h, 0xe000, v161.h
	v_add_nc_u16 v162.l, 0xe000, v162.l
	v_add_nc_u16 v162.h, 0xe000, v162.h
	v_add_nc_u16 v163.l, 0xe000, v163.l
	v_add_nc_u16 v163.h, 0xe000, v163.h
	v_add_nc_u16 v164.l, 0xe000, v164.l
	v_add_nc_u16 v164.h, 0xe000, v164.h
	v_add_nc_u16 v165.l, 0xe000, v165.l
	v_add_nc_u16 v165.h, 0xe000, v165.h
	v_add_nc_u16 v166.l, 0xe000, v166.l
	v_add_nc_u16 v166.h, 0xe000, v166.h
	v_add_nc_u16 v167.l, 0xe000, v167.l
	v_add_nc_u16 v167.h, 0xe000, v167.h
	v_add_nc_u16 v168.l, 0xe000, v168.l
	v_add_nc_u16 v168.h, 0xe000, v168.h
	v_add_nc_u16 v169.l, 0xe000, v169.l
	v_add_nc_u16 v169.h, 0xe000, v169.h
	v_add_nc_u16 v170.l, 0xe000, v170.l
	v_add_nc_u16 v170.h, 0xe000, v170.h
	ds_store_2addr_b32 v88, v6, v7 offset0:176 offset1:192
	ds_store_2addr_b32 v89, v8, v9 offset0:224 offset1:240
	ds_store_2addr_b32 v90, v10, v111 offset0:16 offset1:32
	ds_store_2addr_b32 v92, v112, v113 offset0:64 offset1:80
	ds_store_2addr_b32 v93, v114, v115 offset0:112 offset1:128
	ds_store_2addr_b32 v94, v116, v117 offset0:160 offset1:176
	ds_store_2addr_b32 v95, v118, v119 offset0:208 offset1:224
	ds_store_2addr_b32 v96, v120, v156 offset1:16
	ds_store_2addr_b32 v98, v157, v158 offset0:48 offset1:64
	ds_store_2addr_b32 v99, v159, v160 offset0:96 offset1:112
	ds_store_2addr_b32 v100, v161, v162 offset0:144 offset1:160
	ds_store_2addr_b32 v101, v163, v164 offset0:192 offset1:208
	ds_store_2addr_b32 v102, v165, v166 offset0:112 offset1:128
	ds_store_2addr_b32 v103, v167, v168 offset0:32 offset1:48
	ds_store_2addr_b32 v104, v169, v170 offset0:80 offset1:96
	ds_store_b32 v82, v205 offset:19200
	s_waitcnt vmcnt(37)
	ds_store_b32 v85, v3 offset:19204
	s_waitcnt vmcnt(36)
	ds_store_b32 v86, v4 offset:19204
	s_waitcnt vmcnt(34)
	ds_store_2addr_stride64_b32 v79, v5, v121 offset0:2 offset1:4
	s_waitcnt vmcnt(32)
	ds_store_2addr_stride64_b32 v79, v122, v123 offset0:6 offset1:8
	s_waitcnt vmcnt(30)
	ds_store_2addr_stride64_b32 v79, v124, v125 offset0:10 offset1:12
	s_waitcnt vmcnt(28)
	ds_store_2addr_stride64_b32 v79, v126, v127 offset0:14 offset1:16
	s_waitcnt vmcnt(11)
	ds_store_2addr_stride64_b32 v79, v153, v128 offset0:18 offset1:20
	ds_store_2addr_stride64_b32 v79, v129, v130 offset0:22 offset1:24
	ds_store_2addr_stride64_b32 v79, v131, v132 offset0:26 offset1:28
	ds_store_2addr_stride64_b32 v79, v133, v134 offset0:30 offset1:32
	s_waitcnt vmcnt(10)
	ds_store_2addr_stride64_b32 v79, v146, v135 offset0:34 offset1:36
	ds_store_2addr_stride64_b32 v79, v136, v137 offset0:38 offset1:40
	ds_store_2addr_stride64_b32 v79, v138, v139 offset0:42 offset1:44
	ds_store_2addr_stride64_b32 v79, v140, v141 offset0:46 offset1:48
	s_waitcnt vmcnt(9)
	ds_store_2addr_stride64_b32 v79, v145, v142 offset0:50 offset1:52
	s_waitcnt vmcnt(7)
	ds_store_2addr_stride64_b32 v79, v143, v147 offset0:54 offset1:56
	s_waitcnt vmcnt(5)
	ds_store_2addr_stride64_b32 v79, v148, v149 offset0:58 offset1:60
	s_waitcnt vmcnt(3)
	ds_store_2addr_stride64_b32 v79, v150, v151 offset0:62 offset1:64
	s_waitcnt vmcnt(2)
	ds_store_2addr_stride64_b32 v79, v144, v152 offset0:66 offset1:68
	s_waitcnt vmcnt(0)
	ds_store_2addr_stride64_b32 v79, v154, v155 offset0:70 offset1:72
	v_dual_mov_b32 v111, v78 :: v_dual_mov_b32 v112, v75
	v_mov_b32_e32 v113, v77
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_2addr_b32 v[3:4], v106 offset0:64 offset1:216
	ds_load_2addr_b32 v[5:6], v107 offset0:48 offset1:200
	ds_load_2addr_b32 v[7:8], v108 offset0:32 offset1:184
	ds_load_2addr_b32 v[9:10], v109 offset0:80 offset1:232
.LBB0_2:                                ; %.preheader8.i.i.i
                                        ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v121, s11 :: v_dual_add_nc_u32 v122, 0x4a00, v112
	v_dual_mov_b32 v114, s4 :: v_dual_add_nc_u32 v123, 0xb10, v113
	v_add_nc_u32_e32 v124, 0x1410, v113
	v_add_nc_u32_e32 v125, 0x1d10, v113
	v_add_nc_u32_e32 v126, 0x2610, v113
	ds_load_2addr_b64 v[130:133], v113 offset0:66 offset1:67
	v_add_nc_u32_e32 v127, 0x2f10, v113
	v_add_nc_u32_e32 v128, 0x3810, v113
	v_add_nc_u32_e32 v129, 0x4110, v113
	ds_load_2addr_b64 v[186:189], v122 offset1:1
	ds_load_2addr_b64 v[138:141], v123 offset1:1
	ds_load_2addr_b64 v[146:149], v124 offset1:1
	ds_load_2addr_b64 v[154:157], v125 offset1:1
	ds_load_2addr_b64 v[162:165], v126 offset1:1
	ds_load_2addr_b64 v[170:173], v127 offset1:1
	ds_load_2addr_b64 v[178:181], v128 offset1:1
	ds_load_2addr_b64 v[190:193], v129 offset1:1
	ds_load_i8 v202, v111 offset:19204
	ds_load_i8 v203, v111 offset:19812
	ds_load_i8 v204, v111 offset:20420
	ds_load_i8 v205, v111 offset:21028
	ds_load_i8 v206, v111 offset:21636
	ds_load_i8 v207, v111 offset:22244
	ds_load_i8 v208, v111 offset:22852
	ds_load_i8 v209, v111 offset:23460
	s_add_i32 s18, s18, 4
	v_dual_mov_b32 v120, s10 :: v_dual_mov_b32 v119, s9
	s_lshr_b32 s19, s18, 1
	v_dual_mov_b32 v118, s8 :: v_dual_mov_b32 v117, s7
	s_and_b32 s19, s19, 0x7ffffffc
	v_dual_mov_b32 v116, s6 :: v_dual_mov_b32 v115, s5
	v_add_nc_u32_e32 v122, s19, v77
	ds_load_2addr_stride64_b32 v[194:195], v122 offset0:2 offset1:11
	ds_load_2addr_stride64_b32 v[196:197], v122 offset0:20 offset1:29
	ds_load_2addr_stride64_b32 v[198:199], v122 offset0:38 offset1:47
	ds_load_2addr_stride64_b32 v[200:201], v122 offset0:56 offset1:65
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu8 v[122:129], v[186:189], v[130:133], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu8 v[130:137], v[186:189], v[138:141], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(17)
	v_wmma_i32_16x16x16_iu8 v[138:145], v[186:189], v[146:149], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(16)
	v_wmma_i32_16x16x16_iu8 v[146:153], v[186:189], v[154:157], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(15)
	v_wmma_i32_16x16x16_iu8 v[154:161], v[186:189], v[162:165], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(14)
	v_wmma_i32_16x16x16_iu8 v[162:169], v[186:189], v[170:173], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(13)
	v_wmma_i32_16x16x16_iu8 v[170:177], v[186:189], v[178:181], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu8 v[178:185], v[186:189], v[190:193], v[114:121] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_mul_lo_u32 v114, v122, v202
	s_waitcnt lgkmcnt(10)
	v_mul_lo_u32 v115, v123, v203
	s_waitcnt lgkmcnt(9)
	v_mul_lo_u32 v116, v124, v204
	s_waitcnt lgkmcnt(8)
	v_mul_lo_u32 v117, v125, v205
	s_waitcnt lgkmcnt(7)
	v_mul_lo_u32 v118, v126, v206
	s_waitcnt lgkmcnt(6)
	v_mul_lo_u32 v119, v127, v207
	s_waitcnt lgkmcnt(5)
	v_mul_lo_u32 v120, v128, v208
	s_waitcnt lgkmcnt(4)
	v_mul_lo_u32 v121, v129, v209
	v_mul_lo_u32 v122, v130, v202
	v_mul_lo_u32 v123, v131, v203
	v_mul_lo_u32 v124, v132, v204
	v_mul_lo_u32 v125, v133, v205
	v_mul_lo_u32 v126, v134, v206
	v_mul_lo_u32 v127, v135, v207
	v_mul_lo_u32 v128, v136, v208
	v_mul_lo_u32 v129, v137, v209
	v_mul_lo_u32 v130, v138, v202
	v_mul_lo_u32 v131, v139, v203
	v_mul_lo_u32 v132, v140, v204
	v_mul_lo_u32 v133, v141, v205
	v_mul_lo_u32 v134, v142, v206
	v_mul_lo_u32 v135, v143, v207
	v_mul_lo_u32 v136, v144, v208
	v_mul_lo_u32 v137, v145, v209
	v_mul_lo_u32 v138, v146, v202
	v_mul_lo_u32 v139, v147, v203
	v_mul_lo_u32 v140, v148, v204
	v_mul_lo_u32 v141, v149, v205
	v_mul_lo_u32 v142, v150, v206
	v_mul_lo_u32 v143, v151, v207
	v_mul_lo_u32 v144, v152, v208
	v_mul_lo_u32 v145, v153, v209
	v_mul_lo_u32 v146, v154, v202
	v_mul_lo_u32 v147, v155, v203
	v_mul_lo_u32 v148, v156, v204
	v_mul_lo_u32 v149, v157, v205
	v_mul_lo_u32 v150, v158, v206
	v_mul_lo_u32 v151, v159, v207
	v_mul_lo_u32 v152, v160, v208
	v_mul_lo_u32 v153, v161, v209
	v_mul_lo_u32 v154, v162, v202
	v_mul_lo_u32 v155, v163, v203
	v_mul_lo_u32 v156, v164, v204
	v_mul_lo_u32 v157, v165, v205
	v_mul_lo_u32 v158, v166, v206
	v_mul_lo_u32 v159, v167, v207
	v_mul_lo_u32 v160, v168, v208
	v_mul_lo_u32 v161, v169, v209
	v_mul_lo_u32 v162, v170, v202
	v_mul_lo_u32 v163, v171, v203
	v_mul_lo_u32 v164, v172, v204
	v_mul_lo_u32 v165, v173, v205
	v_mul_lo_u32 v166, v174, v206
	v_mul_lo_u32 v167, v175, v207
	v_mul_lo_u32 v168, v176, v208
	v_mul_lo_u32 v169, v177, v209
	v_mul_lo_u32 v170, v178, v202
	v_mul_lo_u32 v171, v179, v203
	v_mul_lo_u32 v172, v180, v204
	v_mul_lo_u32 v173, v181, v205
	v_mul_lo_u32 v174, v182, v206
	v_mul_lo_u32 v175, v183, v207
	v_mul_lo_u32 v176, v184, v208
	v_mul_lo_u32 v177, v185, v209
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v117, v117
	v_cvt_f32_i32_e32 v118, v118
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v126, v126
	v_cvt_f32_i32_e32 v127, v127
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v130, v130
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v133, v133
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v135, v135
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v140, v140
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v143, v143
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v146, v146
	v_cvt_f32_i32_e32 v147, v147
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v150, v150
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v153, v153
	v_cvt_f32_i32_e32 v154, v154
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v156, v156
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v163, v163
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v170, v170
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v173, v173
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v175, v175
	v_cvt_f32_i32_e32 v176, v176
	v_cvt_f32_i32_e32 v177, v177
	v_dual_mul_f32 v120, v9, v120 :: v_dual_add_nc_u32 v113, 16, v113
	v_dual_mul_f32 v117, v6, v117 :: v_dual_add_nc_u32 v112, 16, v112
	v_dual_mul_f32 v122, v3, v122 :: v_dual_add_nc_u32 v111, 1, v111
	v_dual_mul_f32 v114, v3, v114 :: v_dual_mul_f32 v115, v4, v115
	v_mul_f32_e32 v116, v5, v116
	v_dual_mul_f32 v118, v7, v118 :: v_dual_mul_f32 v119, v8, v119
	v_dual_mul_f32 v121, v10, v121 :: v_dual_mul_f32 v124, v5, v124
	v_mul_f32_e32 v123, v4, v123
	v_dual_mul_f32 v125, v6, v125 :: v_dual_mul_f32 v126, v7, v126
	v_dual_mul_f32 v127, v8, v127 :: v_dual_mul_f32 v128, v9, v128
	v_dual_mul_f32 v129, v10, v129 :: v_dual_mul_f32 v130, v3, v130
	v_dual_mul_f32 v131, v4, v131 :: v_dual_mul_f32 v132, v5, v132
	v_dual_mul_f32 v133, v6, v133 :: v_dual_mul_f32 v134, v7, v134
	v_dual_mul_f32 v135, v8, v135 :: v_dual_mul_f32 v136, v9, v136
	v_dual_mul_f32 v137, v10, v137 :: v_dual_mul_f32 v138, v3, v138
	v_dual_mul_f32 v139, v4, v139 :: v_dual_mul_f32 v140, v5, v140
	v_dual_mul_f32 v141, v6, v141 :: v_dual_mul_f32 v142, v7, v142
	v_dual_mul_f32 v143, v8, v143 :: v_dual_mul_f32 v144, v9, v144
	v_dual_mul_f32 v145, v10, v145 :: v_dual_mul_f32 v146, v3, v146
	v_dual_mul_f32 v147, v4, v147 :: v_dual_mul_f32 v148, v5, v148
	v_dual_mul_f32 v149, v6, v149 :: v_dual_mul_f32 v150, v7, v150
	v_dual_mul_f32 v151, v8, v151 :: v_dual_mul_f32 v152, v9, v152
	v_dual_mul_f32 v153, v10, v153 :: v_dual_mul_f32 v154, v3, v154
	v_dual_mul_f32 v155, v4, v155 :: v_dual_mul_f32 v156, v5, v156
	v_dual_mul_f32 v157, v6, v157 :: v_dual_mul_f32 v158, v7, v158
	v_dual_mul_f32 v159, v8, v159 :: v_dual_mul_f32 v160, v9, v160
	v_dual_mul_f32 v161, v10, v161 :: v_dual_mul_f32 v162, v3, v162
	v_dual_mul_f32 v163, v4, v163 :: v_dual_mul_f32 v164, v5, v164
	v_dual_mul_f32 v165, v6, v165 :: v_dual_mul_f32 v166, v7, v166
	v_dual_mul_f32 v167, v8, v167 :: v_dual_mul_f32 v168, v9, v168
	v_dual_mul_f32 v169, v10, v169 :: v_dual_mul_f32 v170, v3, v170
	v_dual_mul_f32 v171, v4, v171 :: v_dual_mul_f32 v172, v5, v172
	v_dual_mul_f32 v173, v6, v173 :: v_dual_mul_f32 v174, v7, v174
	v_dual_mul_f32 v175, v8, v175 :: v_dual_mul_f32 v176, v9, v176
	v_mul_f32_e32 v177, v10, v177
	s_waitcnt lgkmcnt(3)
	v_dual_fmac_f32 v110, v194, v114 :: v_dual_fmac_f32 v65, v195, v124
	v_dual_fmac_f32 v105, v194, v115 :: v_dual_fmac_f32 v64, v195, v125
	v_dual_fmac_f32 v97, v194, v116 :: v_dual_fmac_f32 v66, v195, v123
	v_dual_fmac_f32 v91, v194, v117 :: v_dual_fmac_f32 v62, v195, v127
	v_dual_fmac_f32 v84, v194, v118 :: v_dual_fmac_f32 v59, v195, v129
	v_dual_fmac_f32 v76, v194, v119 :: v_dual_fmac_f32 v67, v195, v122
	v_dual_fmac_f32 v72, v194, v120 :: v_dual_fmac_f32 v63, v195, v126
	v_dual_fmac_f32 v71, v194, v121 :: v_dual_fmac_f32 v60, v195, v128
	s_waitcnt lgkmcnt(2)
	v_dual_fmac_f32 v58, v196, v130 :: v_dual_fmac_f32 v49, v197, v139
	v_dual_fmac_f32 v57, v196, v131 :: v_dual_fmac_f32 v50, v197, v138
	v_dual_fmac_f32 v56, v196, v132 :: v_dual_fmac_f32 v47, v197, v141
	v_dual_fmac_f32 v55, v196, v133 :: v_dual_fmac_f32 v48, v197, v140
	v_dual_fmac_f32 v54, v196, v134 :: v_dual_fmac_f32 v45, v197, v143
	v_dual_fmac_f32 v53, v196, v135 :: v_dual_fmac_f32 v46, v197, v142
	v_dual_fmac_f32 v52, v196, v136 :: v_dual_fmac_f32 v43, v197, v145
	v_dual_fmac_f32 v51, v196, v137 :: v_dual_fmac_f32 v44, v197, v144
	s_waitcnt lgkmcnt(1)
	v_dual_fmac_f32 v42, v198, v146 :: v_dual_fmac_f32 v33, v199, v155
	v_dual_fmac_f32 v41, v198, v147 :: v_dual_fmac_f32 v34, v199, v154
	v_dual_fmac_f32 v40, v198, v148 :: v_dual_fmac_f32 v31, v199, v157
	v_dual_fmac_f32 v39, v198, v149 :: v_dual_fmac_f32 v32, v199, v156
	v_dual_fmac_f32 v38, v198, v150 :: v_dual_fmac_f32 v29, v199, v159
	v_dual_fmac_f32 v37, v198, v151 :: v_dual_fmac_f32 v30, v199, v158
	v_dual_fmac_f32 v36, v198, v152 :: v_dual_fmac_f32 v27, v199, v161
	v_dual_fmac_f32 v35, v198, v153 :: v_dual_fmac_f32 v28, v199, v160
	s_waitcnt lgkmcnt(0)
	v_dual_fmac_f32 v26, v200, v162 :: v_dual_fmac_f32 v17, v201, v171
	v_dual_fmac_f32 v25, v200, v163 :: v_dual_fmac_f32 v18, v201, v170
	v_dual_fmac_f32 v24, v200, v164 :: v_dual_fmac_f32 v15, v201, v173
	v_dual_fmac_f32 v23, v200, v165 :: v_dual_fmac_f32 v16, v201, v172
	v_dual_fmac_f32 v22, v200, v166 :: v_dual_fmac_f32 v13, v201, v175
	v_dual_fmac_f32 v21, v200, v167 :: v_dual_fmac_f32 v14, v201, v174
	v_dual_fmac_f32 v20, v200, v168 :: v_dual_fmac_f32 v11, v201, v177
	v_dual_fmac_f32 v19, v200, v169 :: v_dual_fmac_f32 v12, v201, v176
	s_cmp_lt_u32 s18, 28
	s_cbranch_scc1 .LBB0_2
; %bb.3:                                ; %_ZL18mmq_vec_dot_targetIL9ggml_type14ELi128ELb1ELb0EEvPKiS2_Pfi.exit.i
                                        ;   in Loop: Header=BB0_1 Depth=1
	v_add_nc_u32_e32 v2, s22, v2
	s_barrier
	buffer_gl0_inv
	s_mov_b32 s18, -4
	v_ashrrev_i32_e32 v3, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[2:3], 2, v[2:3]
	v_add_co_u32 v2, vcc_lo, s14, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	s_clause 0x7
	global_load_b32 v10, v[2:3], off
	global_load_b32 v113, v[2:3], off offset:512
	global_load_b32 v114, v[2:3], off offset:1024
	global_load_b32 v115, v[2:3], off offset:1536
	global_load_b32 v116, v[2:3], off offset:2048
	global_load_b32 v117, v[2:3], off offset:2560
	global_load_b32 v118, v[2:3], off offset:3072
	global_load_b32 v119, v[2:3], off offset:3584
	v_add_co_u32 v4, vcc_lo, 0x1000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, 0, v3, vcc_lo
	v_add_co_u32 v6, vcc_lo, v2, 0x2000
	v_add_co_ci_u32_e64 v7, null, 0, v3, vcc_lo
	v_add_co_u32 v8, vcc_lo, 0x2000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v9, null, 0, v3, vcc_lo
	s_clause 0x7
	global_load_b32 v120, v[4:5], off offset:512
	global_load_b32 v121, v[4:5], off offset:1024
	global_load_b32 v122, v[4:5], off offset:1536
	global_load_b32 v123, v[4:5], off offset:2048
	global_load_b32 v124, v[4:5], off offset:2560
	global_load_b32 v125, v[4:5], off offset:3072
	global_load_b32 v126, v[4:5], off offset:3584
	global_load_b32 v127, v[8:9], off offset:512
	v_add_co_u32 v4, vcc_lo, 0x3000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, 0, v3, vcc_lo
	v_add_co_u32 v111, vcc_lo, v2, 0x4000
	v_add_co_ci_u32_e64 v112, null, 0, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, 0x4000, v2
	s_clause 0x7
	global_load_b32 v128, v[8:9], off offset:1024
	global_load_b32 v129, v[8:9], off offset:1536
	global_load_b32 v130, v[8:9], off offset:2048
	global_load_b32 v131, v[8:9], off offset:2560
	global_load_b32 v132, v[8:9], off offset:3072
	global_load_b32 v8, v[8:9], off offset:3584
	global_load_b32 v9, v[4:5], off offset:512
	global_load_b32 v133, v[4:5], off offset:1024
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_clause 0xb
	global_load_b32 v134, v[6:7], off offset:-4096
	global_load_b32 v6, v[6:7], off
	global_load_b32 v7, v[111:112], off offset:-4096
	global_load_b32 v111, v[111:112], off
	global_load_b32 v112, v[4:5], off offset:1536
	global_load_b32 v135, v[4:5], off offset:2048
	global_load_b32 v136, v[4:5], off offset:2560
	global_load_b32 v137, v[4:5], off offset:3072
	global_load_b32 v4, v[4:5], off offset:3584
	global_load_b32 v5, v[2:3], off offset:512
	global_load_b32 v138, v[2:3], off offset:1024
	global_load_b32 v2, v[2:3], off offset:1536
	s_waitcnt vmcnt(34)
	ds_store_2addr_stride64_b32 v79, v10, v113 offset0:2 offset1:4
	s_waitcnt vmcnt(32)
	ds_store_2addr_stride64_b32 v79, v114, v115 offset0:6 offset1:8
	s_waitcnt vmcnt(30)
	ds_store_2addr_stride64_b32 v79, v116, v117 offset0:10 offset1:12
	s_waitcnt vmcnt(28)
	ds_store_2addr_stride64_b32 v79, v118, v119 offset0:14 offset1:16
	s_waitcnt vmcnt(11)
	ds_store_2addr_stride64_b32 v79, v134, v120 offset0:18 offset1:20
	ds_store_2addr_stride64_b32 v79, v121, v122 offset0:22 offset1:24
	ds_store_2addr_stride64_b32 v79, v123, v124 offset0:26 offset1:28
	ds_store_2addr_stride64_b32 v79, v125, v126 offset0:30 offset1:32
	s_waitcnt vmcnt(10)
	ds_store_2addr_stride64_b32 v79, v6, v127 offset0:34 offset1:36
	ds_store_2addr_stride64_b32 v79, v128, v129 offset0:38 offset1:40
	ds_store_2addr_stride64_b32 v79, v130, v131 offset0:42 offset1:44
	ds_store_2addr_stride64_b32 v79, v132, v8 offset0:46 offset1:48
	s_waitcnt vmcnt(9)
	ds_store_2addr_stride64_b32 v79, v7, v9 offset0:50 offset1:52
	s_waitcnt vmcnt(7)
	ds_store_2addr_stride64_b32 v79, v133, v112 offset0:54 offset1:56
	s_waitcnt vmcnt(5)
	ds_store_2addr_stride64_b32 v79, v135, v136 offset0:58 offset1:60
	s_waitcnt vmcnt(3)
	ds_store_2addr_stride64_b32 v79, v137, v4 offset0:62 offset1:64
	s_waitcnt vmcnt(2)
	ds_store_2addr_stride64_b32 v79, v111, v5 offset0:66 offset1:68
	s_waitcnt vmcnt(0)
	ds_store_2addr_stride64_b32 v79, v138, v2 offset0:70 offset1:72
	v_dual_mov_b32 v10, v78 :: v_dual_mov_b32 v111, v75
	v_mov_b32_e32 v112, v77
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_2addr_b32 v[2:3], v106 offset0:64 offset1:216
	ds_load_2addr_b32 v[4:5], v107 offset0:48 offset1:200
	ds_load_2addr_b32 v[6:7], v108 offset0:32 offset1:184
	ds_load_2addr_b32 v[8:9], v109 offset0:80 offset1:232
.LBB0_4:                                ; %.preheader8.i.i125.i
                                        ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_dual_mov_b32 v120, s11 :: v_dual_add_nc_u32 v121, 0x4a80, v111
	v_dual_mov_b32 v113, s4 :: v_dual_add_nc_u32 v122, 0xb10, v112
	v_add_nc_u32_e32 v123, 0x1410, v112
	v_add_nc_u32_e32 v124, 0x1d10, v112
	v_add_nc_u32_e32 v125, 0x2610, v112
	ds_load_2addr_b64 v[129:132], v112 offset0:66 offset1:67
	v_add_nc_u32_e32 v126, 0x2f10, v112
	v_add_nc_u32_e32 v127, 0x3810, v112
	v_add_nc_u32_e32 v128, 0x4110, v112
	ds_load_2addr_b64 v[185:188], v121 offset1:1
	ds_load_2addr_b64 v[137:140], v122 offset1:1
	ds_load_2addr_b64 v[145:148], v123 offset1:1
	ds_load_2addr_b64 v[153:156], v124 offset1:1
	ds_load_2addr_b64 v[161:164], v125 offset1:1
	ds_load_2addr_b64 v[169:172], v126 offset1:1
	ds_load_2addr_b64 v[177:180], v127 offset1:1
	ds_load_2addr_b64 v[189:192], v128 offset1:1
	ds_load_i8 v201, v10 offset:19212
	ds_load_i8 v202, v10 offset:19820
	ds_load_i8 v203, v10 offset:20428
	ds_load_i8 v204, v10 offset:21036
	ds_load_i8 v205, v10 offset:21644
	ds_load_i8 v206, v10 offset:22252
	ds_load_i8 v207, v10 offset:22860
	ds_load_i8 v208, v10 offset:23468
	s_add_i32 s18, s18, 4
	v_dual_mov_b32 v119, s10 :: v_dual_mov_b32 v118, s9
	s_lshr_b32 s19, s18, 1
	v_dual_mov_b32 v117, s8 :: v_dual_mov_b32 v116, s7
	s_and_b32 s19, s19, 0x7ffffffc
	v_dual_mov_b32 v115, s6 :: v_dual_mov_b32 v114, s5
	v_add_nc_u32_e32 v121, s19, v77
	ds_load_2addr_stride64_b32 v[193:194], v121 offset0:2 offset1:11
	ds_load_2addr_stride64_b32 v[195:196], v121 offset0:20 offset1:29
	ds_load_2addr_stride64_b32 v[197:198], v121 offset0:38 offset1:47
	ds_load_2addr_stride64_b32 v[199:200], v121 offset0:56 offset1:65
	s_waitcnt lgkmcnt(19)
	v_wmma_i32_16x16x16_iu8 v[121:128], v[185:188], v[129:132], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(18)
	v_wmma_i32_16x16x16_iu8 v[129:136], v[185:188], v[137:140], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(17)
	v_wmma_i32_16x16x16_iu8 v[137:144], v[185:188], v[145:148], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(16)
	v_wmma_i32_16x16x16_iu8 v[145:152], v[185:188], v[153:156], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(15)
	v_wmma_i32_16x16x16_iu8 v[153:160], v[185:188], v[161:164], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(14)
	v_wmma_i32_16x16x16_iu8 v[161:168], v[185:188], v[169:172], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(13)
	v_wmma_i32_16x16x16_iu8 v[169:176], v[185:188], v[177:180], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(12)
	v_wmma_i32_16x16x16_iu8 v[177:184], v[185:188], v[189:192], v[113:120] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(11)
	v_mul_lo_u32 v113, v121, v201
	s_waitcnt lgkmcnt(10)
	v_mul_lo_u32 v114, v122, v202
	s_waitcnt lgkmcnt(9)
	v_mul_lo_u32 v115, v123, v203
	s_waitcnt lgkmcnt(8)
	v_mul_lo_u32 v116, v124, v204
	s_waitcnt lgkmcnt(7)
	v_mul_lo_u32 v117, v125, v205
	s_waitcnt lgkmcnt(6)
	v_mul_lo_u32 v118, v126, v206
	s_waitcnt lgkmcnt(5)
	v_mul_lo_u32 v119, v127, v207
	s_waitcnt lgkmcnt(4)
	v_mul_lo_u32 v120, v128, v208
	v_mul_lo_u32 v121, v129, v201
	v_mul_lo_u32 v122, v130, v202
	v_mul_lo_u32 v123, v131, v203
	v_mul_lo_u32 v124, v132, v204
	v_mul_lo_u32 v125, v133, v205
	v_mul_lo_u32 v126, v134, v206
	v_mul_lo_u32 v127, v135, v207
	v_mul_lo_u32 v128, v136, v208
	v_mul_lo_u32 v129, v137, v201
	v_mul_lo_u32 v130, v138, v202
	v_mul_lo_u32 v131, v139, v203
	v_mul_lo_u32 v132, v140, v204
	v_mul_lo_u32 v133, v141, v205
	v_mul_lo_u32 v134, v142, v206
	v_mul_lo_u32 v135, v143, v207
	v_mul_lo_u32 v136, v144, v208
	v_mul_lo_u32 v137, v145, v201
	v_mul_lo_u32 v138, v146, v202
	v_mul_lo_u32 v139, v147, v203
	v_mul_lo_u32 v140, v148, v204
	v_mul_lo_u32 v141, v149, v205
	v_mul_lo_u32 v142, v150, v206
	v_mul_lo_u32 v143, v151, v207
	v_mul_lo_u32 v144, v152, v208
	v_mul_lo_u32 v145, v153, v201
	v_mul_lo_u32 v146, v154, v202
	v_mul_lo_u32 v147, v155, v203
	v_mul_lo_u32 v148, v156, v204
	v_mul_lo_u32 v149, v157, v205
	v_mul_lo_u32 v150, v158, v206
	v_mul_lo_u32 v151, v159, v207
	v_mul_lo_u32 v152, v160, v208
	v_mul_lo_u32 v153, v161, v201
	v_mul_lo_u32 v154, v162, v202
	v_mul_lo_u32 v155, v163, v203
	v_mul_lo_u32 v156, v164, v204
	v_mul_lo_u32 v157, v165, v205
	v_mul_lo_u32 v158, v166, v206
	v_mul_lo_u32 v159, v167, v207
	v_mul_lo_u32 v160, v168, v208
	v_mul_lo_u32 v161, v169, v201
	v_mul_lo_u32 v162, v170, v202
	v_mul_lo_u32 v163, v171, v203
	v_mul_lo_u32 v164, v172, v204
	v_mul_lo_u32 v165, v173, v205
	v_mul_lo_u32 v166, v174, v206
	v_mul_lo_u32 v167, v175, v207
	v_mul_lo_u32 v168, v176, v208
	v_mul_lo_u32 v169, v177, v201
	v_mul_lo_u32 v170, v178, v202
	v_mul_lo_u32 v171, v179, v203
	v_mul_lo_u32 v172, v180, v204
	v_mul_lo_u32 v173, v181, v205
	v_mul_lo_u32 v174, v182, v206
	v_mul_lo_u32 v175, v183, v207
	v_mul_lo_u32 v176, v184, v208
	v_cvt_f32_i32_e32 v113, v113
	v_cvt_f32_i32_e32 v114, v114
	v_cvt_f32_i32_e32 v115, v115
	v_cvt_f32_i32_e32 v116, v116
	v_cvt_f32_i32_e32 v117, v117
	v_cvt_f32_i32_e32 v118, v118
	v_cvt_f32_i32_e32 v119, v119
	v_cvt_f32_i32_e32 v120, v120
	v_cvt_f32_i32_e32 v121, v121
	v_cvt_f32_i32_e32 v122, v122
	v_cvt_f32_i32_e32 v123, v123
	v_cvt_f32_i32_e32 v124, v124
	v_cvt_f32_i32_e32 v125, v125
	v_cvt_f32_i32_e32 v126, v126
	v_cvt_f32_i32_e32 v127, v127
	v_cvt_f32_i32_e32 v128, v128
	v_cvt_f32_i32_e32 v129, v129
	v_cvt_f32_i32_e32 v130, v130
	v_cvt_f32_i32_e32 v131, v131
	v_cvt_f32_i32_e32 v132, v132
	v_cvt_f32_i32_e32 v133, v133
	v_cvt_f32_i32_e32 v134, v134
	v_cvt_f32_i32_e32 v135, v135
	v_cvt_f32_i32_e32 v136, v136
	v_cvt_f32_i32_e32 v137, v137
	v_cvt_f32_i32_e32 v138, v138
	v_cvt_f32_i32_e32 v139, v139
	v_cvt_f32_i32_e32 v140, v140
	v_cvt_f32_i32_e32 v141, v141
	v_cvt_f32_i32_e32 v142, v142
	v_cvt_f32_i32_e32 v143, v143
	v_cvt_f32_i32_e32 v144, v144
	v_cvt_f32_i32_e32 v145, v145
	v_cvt_f32_i32_e32 v146, v146
	v_cvt_f32_i32_e32 v147, v147
	v_cvt_f32_i32_e32 v148, v148
	v_cvt_f32_i32_e32 v149, v149
	v_cvt_f32_i32_e32 v150, v150
	v_cvt_f32_i32_e32 v151, v151
	v_cvt_f32_i32_e32 v152, v152
	v_cvt_f32_i32_e32 v153, v153
	v_cvt_f32_i32_e32 v154, v154
	v_cvt_f32_i32_e32 v155, v155
	v_cvt_f32_i32_e32 v156, v156
	v_cvt_f32_i32_e32 v157, v157
	v_cvt_f32_i32_e32 v158, v158
	v_cvt_f32_i32_e32 v159, v159
	v_cvt_f32_i32_e32 v160, v160
	v_cvt_f32_i32_e32 v161, v161
	v_cvt_f32_i32_e32 v162, v162
	v_cvt_f32_i32_e32 v163, v163
	v_cvt_f32_i32_e32 v164, v164
	v_cvt_f32_i32_e32 v165, v165
	v_cvt_f32_i32_e32 v166, v166
	v_cvt_f32_i32_e32 v167, v167
	v_cvt_f32_i32_e32 v168, v168
	v_cvt_f32_i32_e32 v169, v169
	v_cvt_f32_i32_e32 v170, v170
	v_cvt_f32_i32_e32 v171, v171
	v_cvt_f32_i32_e32 v172, v172
	v_cvt_f32_i32_e32 v173, v173
	v_cvt_f32_i32_e32 v174, v174
	v_cvt_f32_i32_e32 v175, v175
	v_cvt_f32_i32_e32 v176, v176
	v_dual_mul_f32 v119, v8, v119 :: v_dual_add_nc_u32 v112, 16, v112
	v_dual_mul_f32 v116, v5, v116 :: v_dual_add_nc_u32 v111, 16, v111
	v_dual_mul_f32 v121, v2, v121 :: v_dual_add_nc_u32 v10, 1, v10
	v_dual_mul_f32 v113, v2, v113 :: v_dual_mul_f32 v114, v3, v114
	v_mul_f32_e32 v115, v4, v115
	v_dual_mul_f32 v117, v6, v117 :: v_dual_mul_f32 v118, v7, v118
	v_mul_f32_e32 v120, v9, v120
	v_dual_mul_f32 v122, v3, v122 :: v_dual_mul_f32 v123, v4, v123
	v_dual_mul_f32 v124, v5, v124 :: v_dual_mul_f32 v125, v6, v125
	v_dual_mul_f32 v126, v7, v126 :: v_dual_mul_f32 v127, v8, v127
	v_dual_mul_f32 v128, v9, v128 :: v_dual_mul_f32 v129, v2, v129
	v_dual_mul_f32 v130, v3, v130 :: v_dual_mul_f32 v131, v4, v131
	v_dual_mul_f32 v132, v5, v132 :: v_dual_mul_f32 v133, v6, v133
	v_dual_mul_f32 v134, v7, v134 :: v_dual_mul_f32 v135, v8, v135
	v_dual_mul_f32 v136, v9, v136 :: v_dual_mul_f32 v137, v2, v137
	v_dual_mul_f32 v138, v3, v138 :: v_dual_mul_f32 v139, v4, v139
	v_dual_mul_f32 v140, v5, v140 :: v_dual_mul_f32 v141, v6, v141
	v_dual_mul_f32 v142, v7, v142 :: v_dual_mul_f32 v143, v8, v143
	v_dual_mul_f32 v144, v9, v144 :: v_dual_mul_f32 v145, v2, v145
	v_dual_mul_f32 v146, v3, v146 :: v_dual_mul_f32 v147, v4, v147
	v_dual_mul_f32 v148, v5, v148 :: v_dual_mul_f32 v149, v6, v149
	v_dual_mul_f32 v150, v7, v150 :: v_dual_mul_f32 v151, v8, v151
	v_dual_mul_f32 v152, v9, v152 :: v_dual_mul_f32 v153, v2, v153
	v_dual_mul_f32 v154, v3, v154 :: v_dual_mul_f32 v155, v4, v155
	v_dual_mul_f32 v156, v5, v156 :: v_dual_mul_f32 v157, v6, v157
	v_dual_mul_f32 v158, v7, v158 :: v_dual_mul_f32 v159, v8, v159
	v_dual_mul_f32 v160, v9, v160 :: v_dual_mul_f32 v161, v2, v161
	v_dual_mul_f32 v162, v3, v162 :: v_dual_mul_f32 v163, v4, v163
	v_dual_mul_f32 v164, v5, v164 :: v_dual_mul_f32 v165, v6, v165
	v_dual_mul_f32 v166, v7, v166 :: v_dual_mul_f32 v167, v8, v167
	v_dual_mul_f32 v168, v9, v168 :: v_dual_mul_f32 v169, v2, v169
	v_dual_mul_f32 v170, v3, v170 :: v_dual_mul_f32 v171, v4, v171
	v_dual_mul_f32 v172, v5, v172 :: v_dual_mul_f32 v173, v6, v173
	v_dual_mul_f32 v174, v7, v174 :: v_dual_mul_f32 v175, v8, v175
	v_mul_f32_e32 v176, v9, v176
	s_waitcnt lgkmcnt(3)
	v_dual_fmac_f32 v110, v193, v113 :: v_dual_fmac_f32 v65, v194, v123
	v_dual_fmac_f32 v105, v193, v114 :: v_dual_fmac_f32 v64, v194, v124
	v_dual_fmac_f32 v97, v193, v115 :: v_dual_fmac_f32 v66, v194, v122
	v_dual_fmac_f32 v91, v193, v116 :: v_dual_fmac_f32 v62, v194, v126
	v_dual_fmac_f32 v84, v193, v117 :: v_dual_fmac_f32 v59, v194, v128
	v_dual_fmac_f32 v76, v193, v118 :: v_dual_fmac_f32 v67, v194, v121
	v_dual_fmac_f32 v72, v193, v119 :: v_dual_fmac_f32 v63, v194, v125
	v_dual_fmac_f32 v71, v193, v120 :: v_dual_fmac_f32 v60, v194, v127
	s_waitcnt lgkmcnt(2)
	v_dual_fmac_f32 v58, v195, v129 :: v_dual_fmac_f32 v49, v196, v138
	v_dual_fmac_f32 v57, v195, v130 :: v_dual_fmac_f32 v50, v196, v137
	v_dual_fmac_f32 v56, v195, v131 :: v_dual_fmac_f32 v47, v196, v140
	v_dual_fmac_f32 v55, v195, v132 :: v_dual_fmac_f32 v48, v196, v139
	v_dual_fmac_f32 v54, v195, v133 :: v_dual_fmac_f32 v45, v196, v142
	v_dual_fmac_f32 v53, v195, v134 :: v_dual_fmac_f32 v46, v196, v141
	v_dual_fmac_f32 v52, v195, v135 :: v_dual_fmac_f32 v43, v196, v144
	v_dual_fmac_f32 v51, v195, v136 :: v_dual_fmac_f32 v44, v196, v143
	s_waitcnt lgkmcnt(1)
	v_dual_fmac_f32 v42, v197, v145 :: v_dual_fmac_f32 v33, v198, v154
	v_dual_fmac_f32 v41, v197, v146 :: v_dual_fmac_f32 v34, v198, v153
	v_dual_fmac_f32 v40, v197, v147 :: v_dual_fmac_f32 v31, v198, v156
	v_dual_fmac_f32 v39, v197, v148 :: v_dual_fmac_f32 v32, v198, v155
	v_dual_fmac_f32 v38, v197, v149 :: v_dual_fmac_f32 v29, v198, v158
	v_dual_fmac_f32 v37, v197, v150 :: v_dual_fmac_f32 v30, v198, v157
	v_dual_fmac_f32 v36, v197, v151 :: v_dual_fmac_f32 v27, v198, v160
	v_dual_fmac_f32 v35, v197, v152 :: v_dual_fmac_f32 v28, v198, v159
	s_waitcnt lgkmcnt(0)
	v_dual_fmac_f32 v26, v199, v161 :: v_dual_fmac_f32 v17, v200, v170
	v_dual_fmac_f32 v25, v199, v162 :: v_dual_fmac_f32 v18, v200, v169
	v_dual_fmac_f32 v24, v199, v163 :: v_dual_fmac_f32 v15, v200, v172
	v_dual_fmac_f32 v23, v199, v164 :: v_dual_fmac_f32 v16, v200, v171
	v_dual_fmac_f32 v22, v199, v165 :: v_dual_fmac_f32 v13, v200, v174
	v_dual_fmac_f32 v21, v199, v166 :: v_dual_fmac_f32 v14, v200, v173
	v_dual_fmac_f32 v20, v199, v167 :: v_dual_fmac_f32 v11, v200, v176
	v_dual_fmac_f32 v19, v199, v168 :: v_dual_fmac_f32 v12, v200, v175
	s_cmp_lt_u32 s18, 28
	s_cbranch_scc1 .LBB0_4
; %bb.5:                                ; %_ZL18mmq_vec_dot_targetIL9ggml_type14ELi128ELb1ELb0EEvPKiS2_Pfi.exit244.i
                                        ;   in Loop: Header=BB0_1 Depth=1
	s_add_i32 s23, s23, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s23, 8
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_1
; %bb.6:                                ; %_ZL19dense_mmq_bf16_bodyIL9ggml_type14ELi128ELi8ELb1ELb1EEvPKcPKiP14__hip_bfloat16iiii.exit
	s_load_b32 s4, s[0:1], 0x18
	v_bfe_u32 v1, v110, 16, 1
	v_or_b32_e32 v2, 0x400000, v110
	v_bfe_u32 v3, v105, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v110, v110
	v_or_b32_e32 v4, 0x400000, v105
	v_add3_u32 v68, v1, v110, 0x7fff
	v_bfe_u32 v5, v97, 16, 1
	v_add3_u32 v3, v3, v105, 0x7fff
	s_lshl_b32 s0, s2, 6
	v_or_b32_e32 v6, 0x400000, v97
	v_bfe_u32 v7, v91, 16, 1
	v_add3_u32 v5, v5, v97, 0x7fff
	v_or_b32_e32 v8, 0x400000, v91
	v_bfe_u32 v9, v84, 16, 1
	v_bfe_u32 v10, v76, 16, 1
	v_add3_u32 v7, v7, v91, 0x7fff
	v_or_b32_e32 v69, 0x400000, v71
	v_bfe_u32 v70, v67, 16, 1
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[0:1], null, s4, v61, v[0:1]
	v_cndmask_b32_e32 v61, v68, v2, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v105, v105
	s_mul_i32 s2, s4, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	s_ashr_i32 s3, s2, 31
	v_cndmask_b32_e32 v68, v3, v4, vcc_lo
	v_ashrrev_i32_e32 v1, 31, v0
	v_cmp_u_f32_e32 vcc_lo, v97, v97
	s_lshl_b64 s[2:3], s[2:3], 1
	v_add3_u32 v3, v9, v84, 0x7fff
	s_add_u32 s2, s16, s2
	s_addc_u32 s3, s17, s3
	s_ashr_i32 s1, s0, 31
	v_lshlrev_b64 v[1:2], 1, v[0:1]
	v_cndmask_b32_e32 v5, v5, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v91, v91
	s_lshl_b64 s[0:1], s[0:1], 1
	v_or_b32_e32 v4, 0x400000, v84
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	v_cndmask_b32_e32 v6, v7, v8, vcc_lo
	v_add_co_u32 v1, vcc_lo, s0, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s1, v2, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v84, v84
	v_add3_u32 v7, v10, v76, 0x7fff
	v_or_b32_e32 v8, 0x400000, v76
	v_bfe_u32 v9, v72, 16, 1
	s_lshl_b32 s2, s4, 4
	v_cndmask_b32_e32 v10, v3, v4, vcc_lo
	v_bfe_u32 v3, v71, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v76, v76
	v_add3_u32 v4, v9, v72, 0x7fff
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_add3_u32 v9, v3, v71, 0x7fff
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_or_b32_e32 v8, 0x400000, v72
	v_cmp_u_f32_e32 vcc_lo, v72, v72
	v_add_nc_u32_e32 v3, s2, v0
	v_cndmask_b32_e32 v8, v4, v8, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v71, v71
	s_delay_alu instid0(VALU_DEP_3)
	v_ashrrev_i32_e32 v4, 31, v3
	v_cndmask_b32_e32 v0, v9, v69, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[1:2], v61, off
	global_store_d16_hi_b16 v[1:2], v68, off offset:4
	global_store_d16_hi_b16 v[1:2], v5, off offset:8
	global_store_d16_hi_b16 v[1:2], v6, off offset:12
	global_store_d16_hi_b16 v[1:2], v10, off offset:16
	global_store_d16_hi_b16 v[1:2], v7, off offset:20
	global_store_d16_hi_b16 v[1:2], v8, off offset:24
	global_store_d16_hi_b16 v[1:2], v0, off offset:28
	v_lshlrev_b64 v[0:1], 1, v[3:4]
	v_add3_u32 v9, v70, v67, 0x7fff
	v_or_b32_e32 v69, 0x400000, v67
	v_cmp_u_f32_e32 vcc_lo, v67, v67
	v_bfe_u32 v70, v66, 16, 1
	v_or_b32_e32 v5, 0x400000, v66
	v_bfe_u32 v6, v65, 16, 1
	v_bfe_u32 v7, v64, 16, 1
	v_cndmask_b32_e32 v4, v9, v69, vcc_lo
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add3_u32 v2, v70, v66, 0x7fff
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v66, v66
	v_add3_u32 v7, v7, v64, 0x7fff
	v_or_b32_e32 v8, 0x400000, v64
	v_bfe_u32 v9, v63, 16, 1
	v_or_b32_e32 v10, 0x400000, v62
	v_cndmask_b32_e32 v5, v2, v5, vcc_lo
	v_add3_u32 v2, v6, v65, 0x7fff
	v_or_b32_e32 v6, 0x400000, v65
	v_cmp_u_f32_e32 vcc_lo, v65, v65
	v_bfe_u32 v61, v60, 16, 1
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e32 v6, v2, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v64, v64
	v_bfe_u32 v2, v62, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v63, 0x7fff
	v_or_b32_e32 v9, 0x400000, v63
	v_cmp_u_f32_e32 vcc_lo, v63, v63
	v_add3_u32 v2, v2, v62, 0x7fff
	v_bfe_u32 v63, v58, 16, 1
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v62, v62
	v_bfe_u32 v9, v59, 16, 1
	v_or_b32_e32 v62, 0x400000, v59
	v_cndmask_b32_e32 v10, v2, v10, vcc_lo
	v_add3_u32 v2, v61, v60, 0x7fff
	v_or_b32_e32 v61, 0x400000, v60
	v_cmp_u_f32_e32 vcc_lo, v60, v60
	v_add3_u32 v9, v9, v59, 0x7fff
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_cndmask_b32_e32 v60, v2, v61, vcc_lo
	v_add_nc_u32_e32 v2, s2, v3
	v_cmp_u_f32_e32 vcc_lo, v59, v59
	v_add3_u32 v59, v63, v58, 0x7fff
	v_or_b32_e32 v61, 0x400000, v58
	v_ashrrev_i32_e32 v3, 31, v2
	v_cndmask_b32_e32 v9, v9, v62, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v4, off
	global_store_d16_hi_b16 v[0:1], v5, off offset:4
	global_store_d16_hi_b16 v[0:1], v6, off offset:8
	global_store_d16_hi_b16 v[0:1], v7, off offset:12
	global_store_d16_hi_b16 v[0:1], v8, off offset:16
	global_store_d16_hi_b16 v[0:1], v10, off offset:20
	global_store_d16_hi_b16 v[0:1], v60, off offset:24
	global_store_d16_hi_b16 v[0:1], v9, off offset:28
	v_cmp_u_f32_e32 vcc_lo, v58, v58
	v_bfe_u32 v62, v57, 16, 1
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_or_b32_e32 v5, 0x400000, v57
	v_bfe_u32 v6, v56, 16, 1
	v_cndmask_b32_e32 v4, v59, v61, vcc_lo
	v_add3_u32 v3, v62, v57, 0x7fff
	v_bfe_u32 v7, v55, 16, 1
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v57, v57
	v_add3_u32 v7, v7, v55, 0x7fff
	v_or_b32_e32 v8, 0x400000, v55
	v_bfe_u32 v9, v54, 16, 1
	v_or_b32_e32 v10, 0x400000, v53
	v_cndmask_b32_e32 v5, v3, v5, vcc_lo
	v_add3_u32 v3, v6, v56, 0x7fff
	v_or_b32_e32 v6, 0x400000, v56
	v_cmp_u_f32_e32 vcc_lo, v56, v56
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v6, v3, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v55, v55
	v_bfe_u32 v3, v53, 16, 1
	v_bfe_u32 v55, v52, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v54, 0x7fff
	v_or_b32_e32 v9, 0x400000, v54
	v_cmp_u_f32_e32 vcc_lo, v54, v54
	v_add3_u32 v3, v3, v53, 0x7fff
	v_or_b32_e32 v54, 0x400000, v51
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v53, v53
	v_bfe_u32 v9, v51, 16, 1
	v_or_b32_e32 v53, 0x400000, v52
	v_cndmask_b32_e32 v10, v3, v10, vcc_lo
	v_add3_u32 v3, v55, v52, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v52, v52
	v_add3_u32 v9, v9, v51, 0x7fff
	v_bfe_u32 v55, v50, 16, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v52, v3, v53, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v51, v51
	v_add3_u32 v51, v55, v50, 0x7fff
	v_or_b32_e32 v53, 0x400000, v50
	v_cndmask_b32_e32 v9, v9, v54, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v4, off
	global_store_d16_hi_b16 v[0:1], v5, off offset:4
	global_store_d16_hi_b16 v[0:1], v6, off offset:8
	global_store_d16_hi_b16 v[0:1], v7, off offset:12
	global_store_d16_hi_b16 v[0:1], v8, off offset:16
	global_store_d16_hi_b16 v[0:1], v10, off offset:20
	global_store_d16_hi_b16 v[0:1], v52, off offset:24
	global_store_d16_hi_b16 v[0:1], v9, off offset:28
	v_bfe_u32 v7, v47, 16, 1
	v_or_b32_e32 v8, 0x400000, v47
	v_cmp_u_f32_e32 vcc_lo, v50, v50
	v_bfe_u32 v54, v49, 16, 1
	v_or_b32_e32 v5, 0x400000, v49
	v_add3_u32 v7, v7, v47, 0x7fff
	v_add_nc_u32_e32 v2, s2, v2
	v_cndmask_b32_e32 v4, v51, v53, vcc_lo
	v_bfe_u32 v6, v48, 16, 1
	v_bfe_u32 v9, v46, 16, 1
	v_or_b32_e32 v10, 0x400000, v45
	v_ashrrev_i32_e32 v3, 31, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_add3_u32 v3, v54, v49, 0x7fff
	v_add_co_u32 v0, vcc_lo, s0, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v49, v49
	v_cndmask_b32_e32 v5, v3, v5, vcc_lo
	v_add3_u32 v3, v6, v48, 0x7fff
	v_or_b32_e32 v6, 0x400000, v48
	v_cmp_u_f32_e32 vcc_lo, v48, v48
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v6, v3, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v47, v47
	v_bfe_u32 v3, v45, 16, 1
	v_bfe_u32 v47, v44, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v46, 0x7fff
	v_or_b32_e32 v9, 0x400000, v46
	v_cmp_u_f32_e32 vcc_lo, v46, v46
	v_add3_u32 v3, v3, v45, 0x7fff
	v_or_b32_e32 v46, 0x400000, v43
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v45, v45
	v_bfe_u32 v9, v43, 16, 1
	v_or_b32_e32 v45, 0x400000, v44
	v_cndmask_b32_e32 v10, v3, v10, vcc_lo
	v_add3_u32 v3, v47, v44, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v44, v44
	v_add3_u32 v9, v9, v43, 0x7fff
	v_bfe_u32 v47, v42, 16, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v44, v3, v45, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v43, v43
	v_add3_u32 v43, v47, v42, 0x7fff
	v_or_b32_e32 v45, 0x400000, v42
	v_cndmask_b32_e32 v9, v9, v46, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v4, off
	global_store_d16_hi_b16 v[0:1], v5, off offset:4
	global_store_d16_hi_b16 v[0:1], v6, off offset:8
	global_store_d16_hi_b16 v[0:1], v7, off offset:12
	global_store_d16_hi_b16 v[0:1], v8, off offset:16
	global_store_d16_hi_b16 v[0:1], v10, off offset:20
	global_store_d16_hi_b16 v[0:1], v44, off offset:24
	global_store_d16_hi_b16 v[0:1], v9, off offset:28
	v_or_b32_e32 v5, 0x400000, v41
	v_add_nc_u32_e32 v2, s2, v2
	v_cmp_u_f32_e32 vcc_lo, v42, v42
	v_bfe_u32 v46, v41, 16, 1
	v_bfe_u32 v6, v40, 16, 1
	v_bfe_u32 v7, v39, 16, 1
	v_ashrrev_i32_e32 v3, 31, v2
	v_cndmask_b32_e32 v4, v43, v45, vcc_lo
	v_or_b32_e32 v8, 0x400000, v39
	v_bfe_u32 v9, v38, 16, 1
	v_add3_u32 v7, v7, v39, 0x7fff
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_add3_u32 v3, v46, v41, 0x7fff
	v_or_b32_e32 v10, 0x400000, v37
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v41, v41
	v_cndmask_b32_e32 v5, v3, v5, vcc_lo
	v_add3_u32 v3, v6, v40, 0x7fff
	v_or_b32_e32 v6, 0x400000, v40
	v_cmp_u_f32_e32 vcc_lo, v40, v40
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v6, v3, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v39, v39
	v_bfe_u32 v3, v37, 16, 1
	v_bfe_u32 v39, v36, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v38, 0x7fff
	v_or_b32_e32 v9, 0x400000, v38
	v_cmp_u_f32_e32 vcc_lo, v38, v38
	v_add3_u32 v3, v3, v37, 0x7fff
	v_or_b32_e32 v38, 0x400000, v35
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v37, v37
	v_bfe_u32 v9, v35, 16, 1
	v_or_b32_e32 v37, 0x400000, v36
	v_cndmask_b32_e32 v10, v3, v10, vcc_lo
	v_add3_u32 v3, v39, v36, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v36, v36
	v_add3_u32 v9, v9, v35, 0x7fff
	v_bfe_u32 v39, v34, 16, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v36, v3, v37, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v35, v35
	v_add3_u32 v35, v39, v34, 0x7fff
	v_or_b32_e32 v37, 0x400000, v34
	v_cndmask_b32_e32 v9, v9, v38, vcc_lo
	s_clause 0x7
	global_store_d16_hi_b16 v[0:1], v4, off
	global_store_d16_hi_b16 v[0:1], v5, off offset:4
	global_store_d16_hi_b16 v[0:1], v6, off offset:8
	global_store_d16_hi_b16 v[0:1], v7, off offset:12
	global_store_d16_hi_b16 v[0:1], v8, off offset:16
	global_store_d16_hi_b16 v[0:1], v10, off offset:20
	global_store_d16_hi_b16 v[0:1], v36, off offset:24
	global_store_d16_hi_b16 v[0:1], v9, off offset:28
	v_or_b32_e32 v5, 0x400000, v33
	v_add_nc_u32_e32 v2, s2, v2
	v_cmp_u_f32_e32 vcc_lo, v34, v34
	v_bfe_u32 v38, v33, 16, 1
	v_bfe_u32 v6, v32, 16, 1
	v_bfe_u32 v7, v31, 16, 1
	v_ashrrev_i32_e32 v3, 31, v2
	v_cndmask_b32_e32 v4, v35, v37, vcc_lo
	v_or_b32_e32 v8, 0x400000, v31
	v_bfe_u32 v9, v30, 16, 1
	v_add3_u32 v7, v7, v31, 0x7fff
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_add3_u32 v3, v38, v33, 0x7fff
	v_or_b32_e32 v10, 0x400000, v29
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v33, v33
	v_cndmask_b32_e32 v5, v3, v5, vcc_lo
	v_add3_u32 v3, v6, v32, 0x7fff
	v_or_b32_e32 v6, 0x400000, v32
	v_cmp_u_f32_e32 vcc_lo, v32, v32
	s_delay_alu instid0(VALU_DEP_2)
	v_cndmask_b32_e32 v6, v3, v6, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v31, v31
	v_bfe_u32 v3, v29, 16, 1
	v_bfe_u32 v31, v28, 16, 1
	v_cndmask_b32_e32 v7, v7, v8, vcc_lo
	v_add3_u32 v8, v9, v30, 0x7fff
	v_or_b32_e32 v9, 0x400000, v30
	v_cmp_u_f32_e32 vcc_lo, v30, v30
	v_add3_u32 v3, v3, v29, 0x7fff
	v_or_b32_e32 v30, 0x400000, v27
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v29, v29
	v_bfe_u32 v9, v27, 16, 1
	v_or_b32_e32 v29, 0x400000, v28
	v_cndmask_b32_e32 v10, v3, v10, vcc_lo
	v_add3_u32 v3, v31, v28, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v28, v28
	v_add3_u32 v9, v9, v27, 0x7fff
	v_bfe_u32 v31, v26, 16, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cndmask_b32_e32 v28, v3, v29, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v27, v27
	v_add3_u32 v27, v31, v26, 0x7fff
	v_or_b32_e32 v29, 0x400000, v26
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
	v_or_b32_e32 v5, 0x400000, v25
	v_add_nc_u32_e32 v2, s2, v2
	v_cmp_u_f32_e32 vcc_lo, v26, v26
	v_bfe_u32 v30, v25, 16, 1
	v_bfe_u32 v6, v24, 16, 1
	v_bfe_u32 v7, v23, 16, 1
	v_ashrrev_i32_e32 v3, 31, v2
	v_cndmask_b32_e32 v4, v27, v29, vcc_lo
	v_or_b32_e32 v8, 0x400000, v23
	v_bfe_u32 v9, v22, 16, 1
	v_add3_u32 v7, v7, v23, 0x7fff
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_add3_u32 v3, v30, v25, 0x7fff
	v_or_b32_e32 v10, 0x400000, v21
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, s0, v0
	v_add_co_ci_u32_e64 v1, null, s1, v1, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v25, v25
	v_cndmask_b32_e32 v5, v3, v5, vcc_lo
	v_add3_u32 v3, v6, v24, 0x7fff
	v_or_b32_e32 v6, 0x400000, v24
	v_cmp_u_f32_e32 vcc_lo, v24, v24
	v_add_nc_u32_e32 v2, s2, v2
	s_delay_alu instid0(VALU_DEP_3)
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
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v8, v8, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v21, v21
	v_bfe_u32 v9, v19, 16, 1
	v_or_b32_e32 v21, 0x400000, v20
	v_cndmask_b32_e32 v10, v3, v10, vcc_lo
	v_add3_u32 v3, v23, v20, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v20, v20
	v_add3_u32 v9, v9, v19, 0x7fff
	v_bfe_u32 v23, v18, 16, 1
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
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
	s_delay_alu instid0(VALU_DEP_4)
	v_cndmask_b32_e32 v4, v4, v5, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v15, v15
	v_add3_u32 v5, v7, v14, 0x7fff
	v_or_b32_e32 v7, 0x400000, v14
	v_or_b32_e32 v15, 0x400000, v11
	v_cndmask_b32_e32 v6, v6, v8, vcc_lo
	v_bfe_u32 v8, v13, 16, 1
	v_cmp_u_f32_e32 vcc_lo, v14, v14
	v_or_b32_e32 v14, 0x400000, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add3_u32 v8, v8, v13, 0x7fff
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_bfe_u32 v7, v11, 16, 1
	v_cndmask_b32_e32 v8, v8, v10, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v12, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
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
