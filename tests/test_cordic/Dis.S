riscv32-unknown-elf-objdump -Mmarch=rv32imcxgap9 -d /home/andrea/PULP-TrainLib-Tutorial/pulp-trainlib/tests/test_cordic/BUILD/PULP/GCC_RISCV//test_cordic/test_cordic

/home/andrea/PULP-TrainLib-Tutorial/pulp-trainlib/tests/test_cordic/BUILD/PULP/GCC_RISCV//test_cordic/test_cordic:     file format elf32-littleriscv


Disassembly of section .vectors:

1c008000 <__irq_vector_base>:
1c008000:	08c0006f          	j	1c00808c <pos_illegal_instr>
1c008004:	08c0006f          	j	1c008090 <pos_no_irq_handler>
1c008008:	0880006f          	j	1c008090 <pos_no_irq_handler>
1c00800c:	0840006f          	j	1c008090 <pos_no_irq_handler>
1c008010:	0800006f          	j	1c008090 <pos_no_irq_handler>
1c008014:	07c0006f          	j	1c008090 <pos_no_irq_handler>
1c008018:	0780006f          	j	1c008090 <pos_no_irq_handler>
1c00801c:	0740006f          	j	1c008090 <pos_no_irq_handler>
1c008020:	0700006f          	j	1c008090 <pos_no_irq_handler>
1c008024:	06c0006f          	j	1c008090 <pos_no_irq_handler>
1c008028:	0680006f          	j	1c008090 <pos_no_irq_handler>
1c00802c:	0640006f          	j	1c008090 <pos_no_irq_handler>
1c008030:	0600006f          	j	1c008090 <pos_no_irq_handler>
1c008034:	05c0006f          	j	1c008090 <pos_no_irq_handler>
1c008038:	0580006f          	j	1c008090 <pos_no_irq_handler>
1c00803c:	0540006f          	j	1c008090 <pos_no_irq_handler>
1c008040:	0500006f          	j	1c008090 <pos_no_irq_handler>
1c008044:	04c0006f          	j	1c008090 <pos_no_irq_handler>
1c008048:	0480006f          	j	1c008090 <pos_no_irq_handler>
1c00804c:	0440006f          	j	1c008090 <pos_no_irq_handler>
1c008050:	0400006f          	j	1c008090 <pos_no_irq_handler>
1c008054:	03c0006f          	j	1c008090 <pos_no_irq_handler>
1c008058:	0380006f          	j	1c008090 <pos_no_irq_handler>
1c00805c:	0340006f          	j	1c008090 <pos_no_irq_handler>
1c008060:	0300006f          	j	1c008090 <pos_no_irq_handler>
1c008064:	02c0006f          	j	1c008090 <pos_no_irq_handler>
1c008068:	0280006f          	j	1c008090 <pos_no_irq_handler>
1c00806c:	0240006f          	j	1c008090 <pos_no_irq_handler>
1c008070:	0200006f          	j	1c008090 <pos_no_irq_handler>
1c008074:	01c0006f          	j	1c008090 <pos_no_irq_handler>
1c008078:	0180006f          	j	1c008090 <pos_no_irq_handler>
1c00807c:	0140006f          	j	1c008090 <pos_no_irq_handler>

1c008080 <_start>:
1c008080:	00006517          	auipc	a0,0x6
1c008084:	54450513          	addi	a0,a0,1348 # 1c00e5c4 <pos_init_entry>
1c008088:	00050067          	jr	a0

1c00808c <pos_illegal_instr>:
1c00808c:	0000006f          	j	1c00808c <pos_illegal_instr>

1c008090 <pos_no_irq_handler>:
1c008090:	30200073          	mret

1c008094 <pos_semihosting_call>:
1c008094:	00100073          	ebreak
1c008098:	00008067          	ret

Disassembly of section .text:

1c00809c <cosf>:
1c00809c:	1101                	addi	sp,sp,-32
1c00809e:	3f491737          	lui	a4,0x3f491
1c0080a2:	ce06                	sw	ra,28(sp)
1c0080a4:	c1f536b3          	p.bclr	a3,a0,0,31
1c0080a8:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x2347fe90>
1c0080ac:	00000593          	li	a1,0
1c0080b0:	02d75663          	ble	a3,a4,1c0080dc <cosf+0x40>
1c0080b4:	7f800737          	lui	a4,0x7f800
1c0080b8:	00e6c763          	blt	a3,a4,1c0080c6 <cosf+0x2a>
1c0080bc:	08a57553          	fsub.s	a0,a0,a0
1c0080c0:	40f2                	lw	ra,28(sp)
1c0080c2:	6105                	addi	sp,sp,32
1c0080c4:	8082                	ret
1c0080c6:	002c                	addi	a1,sp,8
1c0080c8:	2841                	jal	1c008158 <__ieee754_rem_pio2f>
1c0080ca:	fa2537b3          	p.bclr	a5,a0,29,2
1c0080ce:	45b2                	lw	a1,12(sp)
1c0080d0:	4522                	lw	a0,8(sp)
1c0080d2:	0017a763          	p.beqimm	a5,1,1c0080e0 <cosf+0x44>
1c0080d6:	0027ab63          	p.beqimm	a5,2,1c0080ec <cosf+0x50>
1c0080da:	eb99                	bnez	a5,1c0080f0 <cosf+0x54>
1c0080dc:	2ce1                	jal	1c0083b4 <__kernel_cosf>
1c0080de:	b7cd                	j	1c0080c0 <cosf+0x24>
1c0080e0:	4605                	li	a2,1
1c0080e2:	101000ef          	jal	ra,1c0089e2 <__kernel_sinf>
1c0080e6:	20a51553          	fneg.s	a0,a0a0
1c0080ea:	bfd9                	j	1c0080c0 <cosf+0x24>
1c0080ec:	24e1                	jal	1c0083b4 <__kernel_cosf>
1c0080ee:	bfe5                	j	1c0080e6 <cosf+0x4a>
1c0080f0:	4605                	li	a2,1
1c0080f2:	0f1000ef          	jal	ra,1c0089e2 <__kernel_sinf>
1c0080f6:	b7e9                	j	1c0080c0 <cosf+0x24>

1c0080f8 <sinf>:
1c0080f8:	1101                	addi	sp,sp,-32
1c0080fa:	3f491737          	lui	a4,0x3f491
1c0080fe:	ce06                	sw	ra,28(sp)
1c008100:	c1f536b3          	p.bclr	a3,a0,0,31
1c008104:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x2347fe90>
1c008108:	00d74863          	blt	a4,a3,1c008118 <sinf+0x20>
1c00810c:	00000593          	li	a1,0
1c008110:	4601                	li	a2,0
1c008112:	0d1000ef          	jal	ra,1c0089e2 <__kernel_sinf>
1c008116:	a039                	j	1c008124 <sinf+0x2c>
1c008118:	7f800737          	lui	a4,0x7f800
1c00811c:	00e6c763          	blt	a3,a4,1c00812a <sinf+0x32>
1c008120:	08a57553          	fsub.s	a0,a0,a0
1c008124:	40f2                	lw	ra,28(sp)
1c008126:	6105                	addi	sp,sp,32
1c008128:	8082                	ret
1c00812a:	002c                	addi	a1,sp,8
1c00812c:	2035                	jal	1c008158 <__ieee754_rem_pio2f>
1c00812e:	fa2537b3          	p.bclr	a5,a0,29,2
1c008132:	45b2                	lw	a1,12(sp)
1c008134:	4522                	lw	a0,8(sp)
1c008136:	0017a763          	p.beqimm	a5,1,1c008144 <sinf+0x4c>
1c00813a:	0027a763          	p.beqimm	a5,2,1c008148 <sinf+0x50>
1c00813e:	eb99                	bnez	a5,1c008154 <sinf+0x5c>
1c008140:	4605                	li	a2,1
1c008142:	bfc1                	j	1c008112 <sinf+0x1a>
1c008144:	2c85                	jal	1c0083b4 <__kernel_cosf>
1c008146:	bff9                	j	1c008124 <sinf+0x2c>
1c008148:	4605                	li	a2,1
1c00814a:	099000ef          	jal	ra,1c0089e2 <__kernel_sinf>
1c00814e:	20a51553          	fneg.s	a0,a0a0
1c008152:	bfc9                	j	1c008124 <sinf+0x2c>
1c008154:	2485                	jal	1c0083b4 <__kernel_cosf>
1c008156:	bfe5                	j	1c00814e <sinf+0x56>

1c008158 <__ieee754_rem_pio2f>:
1c008158:	1101                	addi	sp,sp,-32
1c00815a:	3f491737          	lui	a4,0x3f491
1c00815e:	cc22                	sw	s0,24(sp)
1c008160:	ce06                	sw	ra,28(sp)
1c008162:	ca26                	sw	s1,20(sp)
1c008164:	c84a                	sw	s2,16(sp)
1c008166:	c1f53433          	p.bclr	s0,a0,0,31
1c00816a:	fd870713          	addi	a4,a4,-40 # 3f490fd8 <__l2_shared_end+0x2347fe90>
1c00816e:	00874763          	blt	a4,s0,1c00817c <__ieee754_rem_pio2f+0x24>
1c008172:	c188                	sw	a0,0(a1)
1c008174:	0005a223          	sw	zero,4(a1)
1c008178:	4501                	li	a0,0
1c00817a:	a0a9                	j	1c0081c4 <__ieee754_rem_pio2f+0x6c>
1c00817c:	4016d737          	lui	a4,0x4016d
1c008180:	be370713          	addi	a4,a4,-1053 # 4016cbe3 <__l2_shared_end+0x2415ba9b>
1c008184:	892a                	mv	s2,a0
1c008186:	0a874163          	blt	a4,s0,1c008228 <__ieee754_rem_pio2f+0xd0>
1c00818a:	1c001737          	lui	a4,0x1c001
1c00818e:	c6043433          	p.bclr	s0,s0,3,0
1c008192:	0b072703          	lw	a4,176(a4) # 1c0010b0 <__clz_tab+0x400>
1c008196:	04a05863          	blez	a0,1c0081e6 <__ieee754_rem_pio2f+0x8e>
1c00819a:	08e577d3          	fsub.s	a5,a0,a4
1c00819e:	3fc91737          	lui	a4,0x3fc91
1c0081a2:	fd070713          	addi	a4,a4,-48 # 3fc90fd0 <__l2_shared_end+0x23c7fe88>
1c0081a6:	02e40563          	beq	s0,a4,1c0081d0 <__ieee754_rem_pio2f+0x78>
1c0081aa:	1c001737          	lui	a4,0x1c001
1c0081ae:	0b472703          	lw	a4,180(a4) # 1c0010b4 <__clz_tab+0x404>
1c0081b2:	08e7f6d3          	fsub.s	a3,a5,a4
1c0081b6:	4505                	li	a0,1
1c0081b8:	08d7f7d3          	fsub.s	a5,a5,a3
1c0081bc:	c194                	sw	a3,0(a1)
1c0081be:	08e7f7d3          	fsub.s	a5,a5,a4
1c0081c2:	c1dc                	sw	a5,4(a1)
1c0081c4:	40f2                	lw	ra,28(sp)
1c0081c6:	4462                	lw	s0,24(sp)
1c0081c8:	44d2                	lw	s1,20(sp)
1c0081ca:	4942                	lw	s2,16(sp)
1c0081cc:	6105                	addi	sp,sp,32
1c0081ce:	8082                	ret
1c0081d0:	1c001737          	lui	a4,0x1c001
1c0081d4:	0b872703          	lw	a4,184(a4) # 1c0010b8 <__clz_tab+0x408>
1c0081d8:	08e7f7d3          	fsub.s	a5,a5,a4
1c0081dc:	1c001737          	lui	a4,0x1c001
1c0081e0:	0bc72703          	lw	a4,188(a4) # 1c0010bc <__clz_tab+0x40c>
1c0081e4:	b7f9                	j	1c0081b2 <__ieee754_rem_pio2f+0x5a>
1c0081e6:	00e577d3          	fadd.s	a5,a0,a4
1c0081ea:	3fc91737          	lui	a4,0x3fc91
1c0081ee:	fd070713          	addi	a4,a4,-48 # 3fc90fd0 <__l2_shared_end+0x23c7fe88>
1c0081f2:	02e40063          	beq	s0,a4,1c008212 <__ieee754_rem_pio2f+0xba>
1c0081f6:	1c001737          	lui	a4,0x1c001
1c0081fa:	0b472703          	lw	a4,180(a4) # 1c0010b4 <__clz_tab+0x404>
1c0081fe:	00e7f6d3          	fadd.s	a3,a5,a4
1c008202:	557d                	li	a0,-1
1c008204:	08d7f7d3          	fsub.s	a5,a5,a3
1c008208:	c194                	sw	a3,0(a1)
1c00820a:	00e7f7d3          	fadd.s	a5,a5,a4
1c00820e:	c1dc                	sw	a5,4(a1)
1c008210:	bf55                	j	1c0081c4 <__ieee754_rem_pio2f+0x6c>
1c008212:	1c001737          	lui	a4,0x1c001
1c008216:	0b872703          	lw	a4,184(a4) # 1c0010b8 <__clz_tab+0x408>
1c00821a:	00e7f7d3          	fadd.s	a5,a5,a4
1c00821e:	1c001737          	lui	a4,0x1c001
1c008222:	0bc72703          	lw	a4,188(a4) # 1c0010bc <__clz_tab+0x40c>
1c008226:	bfe1                	j	1c0081fe <__ieee754_rem_pio2f+0xa6>
1c008228:	43491737          	lui	a4,0x43491
1c00822c:	f8070713          	addi	a4,a4,-128 # 43490f80 <__l2_shared_end+0x2747fe38>
1c008230:	84ae                	mv	s1,a1
1c008232:	0e874f63          	blt	a4,s0,1c008330 <__ieee754_rem_pio2f+0x1d8>
1c008236:	039000ef          	jal	ra,1c008a6e <fabsf>
1c00823a:	1c0016b7          	lui	a3,0x1c001
1c00823e:	1c001737          	lui	a4,0x1c001
1c008242:	87aa                	mv	a5,a0
1c008244:	0c072703          	lw	a4,192(a4) # 1c0010c0 <__clz_tab+0x410>
1c008248:	0886a503          	lw	a0,136(a3) # 1c001088 <__clz_tab+0x3d8>
1c00824c:	46fd                	li	a3,31
1c00824e:	50e7f743          	fmadd.s	a4,a5,a4,a0
1c008252:	c0071553          	fcvt.w.s	a0,a4,rtz
1c008256:	1c001737          	lui	a4,0x1c001
1c00825a:	0b072703          	lw	a4,176(a4) # 1c0010b0 <__clz_tab+0x400>
1c00825e:	d0057653          	fcvt.s.w	a2,a0
1c008262:	78e677cb          	fnmsub.s	a5,a2,a4,a5
1c008266:	1c001737          	lui	a4,0x1c001
1c00826a:	0b472703          	lw	a4,180(a4) # 1c0010b4 <__clz_tab+0x404>
1c00826e:	10e67753          	fmul.s	a4,a2,a4
1c008272:	08e7f5d3          	fsub.s	a1,a5,a4
1c008276:	04a6c263          	blt	a3,a0,1c0082ba <__ieee754_rem_pio2f+0x162>
1c00827a:	fff50693          	addi	a3,a0,-1
1c00827e:	00269893          	slli	a7,a3,0x2
1c008282:	1c0016b7          	lui	a3,0x1c001
1c008286:	86868693          	addi	a3,a3,-1944 # 1c000868 <__DTOR_END__>
1c00828a:	2116f683          	p.lw	a3,a7(a3)
1c00828e:	ce043833          	p.bclr	a6,s0,7,0
1c008292:	02d80463          	beq	a6,a3,1c0082ba <__ieee754_rem_pio2f+0x162>
1c008296:	c08c                	sw	a1,0(s1)
1c008298:	4094                	lw	a3,0(s1)
1c00829a:	08d7f7d3          	fsub.s	a5,a5,a3
1c00829e:	08e7f7d3          	fsub.s	a5,a5,a4
1c0082a2:	c0dc                	sw	a5,4(s1)
1c0082a4:	f20950e3          	bgez	s2,1c0081c4 <__ieee754_rem_pio2f+0x6c>
1c0082a8:	20d696d3          	fneg.s	a3,a3a3
1c0082ac:	c094                	sw	a3,0(s1)
1c0082ae:	20f797d3          	fneg.s	a5,a5a5
1c0082b2:	40a00533          	neg	a0,a0
1c0082b6:	c0dc                	sw	a5,4(s1)
1c0082b8:	b731                	j	1c0081c4 <__ieee754_rem_pio2f+0x6c>
1c0082ba:	0175d693          	srli	a3,a1,0x17
1c0082be:	845d                	srai	s0,s0,0x17
1c0082c0:	ee86b6b3          	p.bclr	a3,a3,23,8
1c0082c4:	40d406b3          	sub	a3,s0,a3
1c0082c8:	4821                	li	a6,8
1c0082ca:	fcd856e3          	ble	a3,a6,1c008296 <__ieee754_rem_pio2f+0x13e>
1c0082ce:	1c001737          	lui	a4,0x1c001
1c0082d2:	0b872703          	lw	a4,184(a4) # 1c0010b8 <__clz_tab+0x408>
1c0082d6:	78e676cb          	fnmsub.s	a3,a2,a4,a5
1c0082da:	08d7f7d3          	fsub.s	a5,a5,a3
1c0082de:	78e6774b          	fnmsub.s	a4,a2,a4,a5
1c0082e2:	1c0017b7          	lui	a5,0x1c001
1c0082e6:	0bc7a783          	lw	a5,188(a5) # 1c0010bc <__clz_tab+0x40c>
1c0082ea:	70f67747          	fmsub.s	a4,a2,a5,a4
1c0082ee:	08e6f5d3          	fsub.s	a1,a3,a4
1c0082f2:	0175d793          	srli	a5,a1,0x17
1c0082f6:	ee87b7b3          	p.bclr	a5,a5,23,8
1c0082fa:	8c1d                	sub	s0,s0,a5
1c0082fc:	47e5                	li	a5,25
1c0082fe:	0087c563          	blt	a5,s0,1c008308 <__ieee754_rem_pio2f+0x1b0>
1c008302:	c08c                	sw	a1,0(s1)
1c008304:	87b6                	mv	a5,a3
1c008306:	bf49                	j	1c008298 <__ieee754_rem_pio2f+0x140>
1c008308:	1c0017b7          	lui	a5,0x1c001
1c00830c:	0c47a703          	lw	a4,196(a5) # 1c0010c4 <__clz_tab+0x414>
1c008310:	68e677cb          	fnmsub.s	a5,a2,a4,a3
1c008314:	08f6f6d3          	fsub.s	a3,a3,a5
1c008318:	68e676cb          	fnmsub.s	a3,a2,a4,a3
1c00831c:	1c001737          	lui	a4,0x1c001
1c008320:	0c872703          	lw	a4,200(a4) # 1c0010c8 <__clz_tab+0x418>
1c008324:	68e67747          	fmsub.s	a4,a2,a4,a3
1c008328:	08e7f6d3          	fsub.s	a3,a5,a4
1c00832c:	c094                	sw	a3,0(s1)
1c00832e:	b7ad                	j	1c008298 <__ieee754_rem_pio2f+0x140>
1c008330:	7f800737          	lui	a4,0x7f800
1c008334:	00e44763          	blt	s0,a4,1c008342 <__ieee754_rem_pio2f+0x1ea>
1c008338:	08a577d3          	fsub.s	a5,a0,a0
1c00833c:	c1dc                	sw	a5,4(a1)
1c00833e:	c19c                	sw	a5,0(a1)
1c008340:	bd25                	j	1c008178 <__ieee754_rem_pio2f+0x20>
1c008342:	41745613          	srai	a2,s0,0x17
1c008346:	f7a60613          	addi	a2,a2,-134
1c00834a:	01761793          	slli	a5,a2,0x17
1c00834e:	8c1d                	sub	s0,s0,a5
1c008350:	c00417d3          	fcvt.w.s	a5,s0,rtz
1c008354:	00000593          	li	a1,0
1c008358:	468d                	li	a3,3
1c00835a:	d007f7d3          	fcvt.s.w	a5,a5
1c00835e:	08f47453          	fsub.s	s0,s0,a5
1c008362:	c23e                	sw	a5,4(sp)
1c008364:	1c0017b7          	lui	a5,0x1c001
1c008368:	0cc7a783          	lw	a5,204(a5) # 1c0010cc <__clz_tab+0x41c>
1c00836c:	10f47453          	fmul.s	s0,s0,a5
1c008370:	c0041753          	fcvt.w.s	a4,s0,rtz
1c008374:	d0077753          	fcvt.s.w	a4,a4
1c008378:	08e47453          	fsub.s	s0,s0,a4
1c00837c:	c43a                	sw	a4,8(sp)
1c00837e:	10f477d3          	fmul.s	a5,s0,a5
1c008382:	c63e                	sw	a5,12(sp)
1c008384:	a0b7a7d3          	feq.s	a5,a5,a1
1c008388:	c791                	beqz	a5,1c008394 <__ieee754_rem_pio2f+0x23c>
1c00838a:	a0b72753          	feq.s	a4,a4,a1
1c00838e:	00173693          	seqz	a3,a4
1c008392:	0685                	addi	a3,a3,1
1c008394:	1c0017b7          	lui	a5,0x1c001
1c008398:	8e878793          	addi	a5,a5,-1816 # 1c0008e8 <two_over_pi>
1c00839c:	4709                	li	a4,2
1c00839e:	85a6                	mv	a1,s1
1c0083a0:	0048                	addi	a0,sp,4
1c0083a2:	28c9                	jal	1c008474 <__kernel_rem_pio2f>
1c0083a4:	e20950e3          	bgez	s2,1c0081c4 <__ieee754_rem_pio2f+0x6c>
1c0083a8:	409c                	lw	a5,0(s1)
1c0083aa:	20f797d3          	fneg.s	a5,a5a5
1c0083ae:	c09c                	sw	a5,0(s1)
1c0083b0:	40dc                	lw	a5,4(s1)
1c0083b2:	bdf5                	j	1c0082ae <__ieee754_rem_pio2f+0x156>

1c0083b4 <__kernel_cosf>:
1c0083b4:	c1f53633          	p.bclr	a2,a0,0,31
1c0083b8:	320007b7          	lui	a5,0x32000
1c0083bc:	1c001837          	lui	a6,0x1c001
1c0083c0:	00f65563          	ble	a5,a2,1c0083ca <__kernel_cosf+0x16>
1c0083c4:	c00517d3          	fcvt.w.s	a5,a0,rtz
1c0083c8:	c3dd                	beqz	a5,1c00846e <__kernel_cosf+0xba>
1c0083ca:	10a577d3          	fmul.s	a5,a0,a0
1c0083ce:	1c001737          	lui	a4,0x1c001
1c0083d2:	08872683          	lw	a3,136(a4) # 1c001088 <__clz_tab+0x3d8>
1c0083d6:	10b57553          	fmul.s	a0,a0,a1
1c0083da:	1c001737          	lui	a4,0x1c001
1c0083de:	1c0015b7          	lui	a1,0x1c001
1c0083e2:	0d85a583          	lw	a1,216(a1) # 1c0010d8 <__clz_tab+0x428>
1c0083e6:	0d472703          	lw	a4,212(a4) # 1c0010d4 <__clz_tab+0x424>
1c0083ea:	10d7f6d3          	fmul.s	a3,a5,a3
1c0083ee:	58e7f743          	fmadd.s	a4,a5,a4,a1
1c0083f2:	1c0015b7          	lui	a1,0x1c001
1c0083f6:	0dc5a583          	lw	a1,220(a1) # 1c0010dc <__clz_tab+0x42c>
1c0083fa:	58f77743          	fmadd.s	a4,a4,a5,a1
1c0083fe:	1c0015b7          	lui	a1,0x1c001
1c008402:	0e05a583          	lw	a1,224(a1) # 1c0010e0 <__clz_tab+0x430>
1c008406:	58f77743          	fmadd.s	a4,a4,a5,a1
1c00840a:	1c0015b7          	lui	a1,0x1c001
1c00840e:	0e45a583          	lw	a1,228(a1) # 1c0010e4 <__clz_tab+0x434>
1c008412:	58f77743          	fmadd.s	a4,a4,a5,a1
1c008416:	1c0015b7          	lui	a1,0x1c001
1c00841a:	0e85a583          	lw	a1,232(a1) # 1c0010e8 <__clz_tab+0x438>
1c00841e:	58f77743          	fmadd.s	a4,a4,a5,a1
1c008422:	10f77753          	fmul.s	a4,a4,a5
1c008426:	50e7f7c7          	fmsub.s	a5,a5,a4,a0
1c00842a:	3e99a737          	lui	a4,0x3e99a
1c00842e:	99970713          	addi	a4,a4,-1639 # 3e999999 <__l2_shared_end+0x22988851>
1c008432:	00c74963          	blt	a4,a2,1c008444 <__kernel_cosf+0x90>
1c008436:	08f6f7d3          	fsub.s	a5,a3,a5
1c00843a:	09082503          	lw	a0,144(a6) # 1c001090 <__clz_tab+0x3e0>
1c00843e:	08f57553          	fsub.s	a0,a0,a5
1c008442:	8082                	ret
1c008444:	3f480737          	lui	a4,0x3f480
1c008448:	00c74e63          	blt	a4,a2,1c008464 <__kernel_cosf+0xb0>
1c00844c:	ff000537          	lui	a0,0xff000
1c008450:	962a                	add	a2,a2,a0
1c008452:	09082503          	lw	a0,144(a6)
1c008456:	08c57553          	fsub.s	a0,a0,a2
1c00845a:	08c6f653          	fsub.s	a2,a3,a2
1c00845e:	08f677d3          	fsub.s	a5,a2,a5
1c008462:	bff1                	j	1c00843e <__kernel_cosf+0x8a>
1c008464:	1c001737          	lui	a4,0x1c001
1c008468:	0d072603          	lw	a2,208(a4) # 1c0010d0 <__clz_tab+0x420>
1c00846c:	b7dd                	j	1c008452 <__kernel_cosf+0x9e>
1c00846e:	09082503          	lw	a0,144(a6)
1c008472:	8082                	ret

1c008474 <__kernel_rem_pio2f>:
1c008474:	7125                	addi	sp,sp,-416
1c008476:	19512223          	sw	s5,388(sp)
1c00847a:	1c001ab7          	lui	s5,0x1c001
1c00847e:	19212823          	sw	s2,400(sp)
1c008482:	19312623          	sw	s3,396(sp)
1c008486:	c00a8913          	addi	s2,s5,-1024 # 1c000c00 <init_jk>
1c00848a:	89ba                	mv	s3,a4
1c00848c:	070a                	slli	a4,a4,0x2
1c00848e:	20e97903          	p.lw	s2,a4(s2)
1c008492:	ffd60e13          	addi	t3,a2,-3
1c008496:	4721                	li	a4,8
1c008498:	02ee4e33          	div	t3,t3,a4
1c00849c:	19612023          	sw	s6,384(sp)
1c0084a0:	17712e23          	sw	s7,380(sp)
1c0084a4:	fff68b93          	addi	s7,a3,-1
1c0084a8:	17912a23          	sw	s9,372(sp)
1c0084ac:	cc2a                	sw	a0,24(sp)
1c0084ae:	8cae                	mv	s9,a1
1c0084b0:	188c                	addi	a1,sp,112
1c0084b2:	18912a23          	sw	s1,404(sp)
1c0084b6:	18112e23          	sw	ra,412(sp)
1c0084ba:	18812c23          	sw	s0,408(sp)
1c0084be:	19412423          	sw	s4,392(sp)
1c0084c2:	17812c23          	sw	s8,376(sp)
1c0084c6:	17a12823          	sw	s10,368(sp)
1c0084ca:	17b12623          	sw	s11,364(sp)
1c0084ce:	01790f33          	add	t5,s2,s7
1c0084d2:	8aae                	mv	s5,a1
1c0084d4:	040e6b33          	p.max	s6,t3,zero
1c0084d8:	001b0713          	addi	a4,s6,1
1c0084dc:	417b0333          	sub	t1,s6,s7
1c0084e0:	070e                	slli	a4,a4,0x3
1c0084e2:	00231513          	slli	a0,t1,0x2
1c0084e6:	40e604b3          	sub	s1,a2,a4
1c0084ea:	953e                	add	a0,a0,a5
1c0084ec:	4601                	li	a2,0
1c0084ee:	02cf5063          	ble	a2,t5,1c00850e <__kernel_rem_pio2f+0x9a>
1c0084f2:	00269713          	slli	a4,a3,0x2
1c0084f6:	9756                	add	a4,a4,s5
1c0084f8:	11010313          	addi	t1,sp,272
1c0084fc:	4501                	li	a0,0
1c0084fe:	5f71                	li	t5,-4
1c008500:	04a94863          	blt	s2,a0,1c008550 <__kernel_rem_pio2f+0xdc>
1c008504:	4862                	lw	a6,24(sp)
1c008506:	00000593          	li	a1,0
1c00850a:	4601                	li	a2,0
1c00850c:	a81d                	j	1c008542 <__kernel_rem_pio2f+0xce>
1c00850e:	00c30833          	add	a6,t1,a2
1c008512:	00000713          	li	a4,0
1c008516:	00084863          	bltz	a6,1c008526 <__kernel_rem_pio2f+0xb2>
1c00851a:	00261713          	slli	a4,a2,0x2
1c00851e:	20e57703          	p.lw	a4,a4(a0)
1c008522:	d0077753          	fcvt.s.w	a4,a4
1c008526:	00e5a22b          	p.sw	a4,4(a1!)
1c00852a:	0605                	addi	a2,a2,1
1c00852c:	b7c9                	j	1c0084ee <__kernel_rem_pio2f+0x7a>
1c00852e:	8fba                	mv	t6,a4
1c008530:	43e60fb3          	p.mac	t6,a2,t5
1c008534:	0048228b          	p.lw	t0,4(a6!)
1c008538:	0605                	addi	a2,a2,1
1c00853a:	ffcfaf83          	lw	t6,-4(t6)
1c00853e:	59f2f5c3          	fmadd.s	a1,t0,t6,a1
1c008542:	fecbd6e3          	ble	a2,s7,1c00852e <__kernel_rem_pio2f+0xba>
1c008546:	00b3222b          	p.sw	a1,4(t1!)
1c00854a:	0505                	addi	a0,a0,1
1c00854c:	0711                	addi	a4,a4,4
1c00854e:	bf4d                	j	1c008500 <__kernel_rem_pio2f+0x8c>
1c008550:	00291a13          	slli	s4,s2,0x2
1c008554:	1a71                	addi	s4,s4,-4
1c008556:	1010                	addi	a2,sp,32
1c008558:	9652                	add	a2,a2,s4
1c00855a:	ce32                	sw	a2,28(sp)
1c00855c:	8dca                	mv	s11,s2
1c00855e:	002d9a13          	slli	s4,s11,0x2
1c008562:	1290                	addi	a2,sp,352
1c008564:	01460733          	add	a4,a2,s4
1c008568:	fb072503          	lw	a0,-80(a4)
1c00856c:	02010f13          	addi	t5,sp,32
1c008570:	0a18                	addi	a4,sp,272
1c008572:	01470333          	add	t1,a4,s4
1c008576:	8ffa                	mv	t6,t5
1c008578:	85ee                	mv	a1,s11
1c00857a:	1371                	addi	t1,t1,-4
1c00857c:	14b04163          	bgtz	a1,1c0086be <__kernel_rem_pio2f+0x24a>
1c008580:	85a6                	mv	a1,s1
1c008582:	c836                	sw	a3,16(sp)
1c008584:	c63e                	sw	a5,12(sp)
1c008586:	ca7a                	sw	t5,20(sp)
1c008588:	2b95                	jal	1c008afc <scalbnf>
1c00858a:	1c001737          	lui	a4,0x1c001
1c00858e:	00070713          	mv	a4,a4
1c008592:	4318                	lw	a4,0(a4)
1c008594:	842a                	mv	s0,a0
1c008596:	10e57553          	fmul.s	a0,a0,a4
1c00859a:	29e9                	jal	1c008a74 <floorf>
1c00859c:	1c001637          	lui	a2,0x1c001
1c0085a0:	0ec60613          	addi	a2,a2,236 # 1c0010ec <__clz_tab+0x43c>
1c0085a4:	4210                	lw	a2,0(a2)
1c0085a6:	46c2                	lw	a3,16(sp)
1c0085a8:	47b2                	lw	a5,12(sp)
1c0085aa:	40c5744b          	fnmsub.s	s0,a0,a2,s0
1c0085ae:	4f52                	lw	t5,20(sp)
1c0085b0:	c0041c53          	fcvt.w.s	s8,s0,rtz
1c0085b4:	d00c7553          	fcvt.s.w	a0,s8
1c0085b8:	08a47453          	fsub.s	s0,s0,a0
1c0085bc:	12905d63          	blez	s1,1c0086f6 <__kernel_rem_pio2f+0x282>
1c0085c0:	fffd8593          	addi	a1,s11,-1
1c0085c4:	1298                	addi	a4,sp,352
1c0085c6:	058a                	slli	a1,a1,0x2
1c0085c8:	95ba                	add	a1,a1,a4
1c0085ca:	ec05a703          	lw	a4,-320(a1)
1c0085ce:	4521                	li	a0,8
1c0085d0:	40950fb3          	sub	t6,a0,s1
1c0085d4:	41f75533          	sra	a0,a4,t6
1c0085d8:	9c2a                	add	s8,s8,a0
1c0085da:	01f51533          	sll	a0,a0,t6
1c0085de:	8f09                	sub	a4,a4,a0
1c0085e0:	ece5a023          	sw	a4,-320(a1)
1c0085e4:	459d                	li	a1,7
1c0085e6:	8d85                	sub	a1,a1,s1
1c0085e8:	40b75d33          	sra	s10,a4,a1
1c0085ec:	05a05363          	blez	s10,1c008632 <__kernel_rem_pio2f+0x1be>
1c0085f0:	0c05                	addi	s8,s8,1
1c0085f2:	4501                	li	a0,0
1c0085f4:	4381                	li	t2,0
1c0085f6:	0ff00f93          	li	t6,255
1c0085fa:	10000293          	li	t0,256
1c0085fe:	13b54163          	blt	a0,s11,1c008720 <__kernel_rem_pio2f+0x2ac>
1c008602:	00905663          	blez	s1,1c00860e <__kernel_rem_pio2f+0x19a>
1c008606:	1414a163          	p.beqimm	s1,1,1c008748 <__kernel_rem_pio2f+0x2d4>
1c00860a:	1424ab63          	p.beqimm	s1,2,1c008760 <__kernel_rem_pio2f+0x2ec>
1c00860e:	022d3263          	p.bneimm	s10,2,1c008632 <__kernel_rem_pio2f+0x1be>
1c008612:	1c0015b7          	lui	a1,0x1c001
1c008616:	0905a503          	lw	a0,144(a1) # 1c001090 <__clz_tab+0x3e0>
1c00861a:	08857453          	fsub.s	s0,a0,s0
1c00861e:	00038a63          	beqz	t2,1c008632 <__kernel_rem_pio2f+0x1be>
1c008622:	85a6                	mv	a1,s1
1c008624:	c836                	sw	a3,16(sp)
1c008626:	c63e                	sw	a5,12(sp)
1c008628:	29d1                	jal	1c008afc <scalbnf>
1c00862a:	08a47453          	fsub.s	s0,s0,a0
1c00862e:	47b2                	lw	a5,12(sp)
1c008630:	46c2                	lw	a3,16(sp)
1c008632:	00000593          	li	a1,0
1c008636:	a0b425d3          	feq.s	a1,s0,a1
1c00863a:	1a058b63          	beqz	a1,1c0087f0 <__kernel_rem_pio2f+0x37c>
1c00863e:	fffd8413          	addi	s0,s11,-1
1c008642:	1018                	addi	a4,sp,32
1c008644:	01470f33          	add	t5,a4,s4
1c008648:	8522                	mv	a0,s0
1c00864a:	4581                	li	a1,0
1c00864c:	1f71                	addi	t5,t5,-4
1c00864e:	13255363          	ble	s2,a0,1c008774 <__kernel_rem_pio2f+0x300>
1c008652:	14058f63          	beqz	a1,1c0087b0 <__kernel_rem_pio2f+0x33c>
1c008656:	00241793          	slli	a5,s0,0x2
1c00865a:	1010                	addi	a2,sp,32
1c00865c:	14e1                	addi	s1,s1,-8
1c00865e:	97b2                	add	a5,a5,a2
1c008660:	ffc7a68b          	p.lw	a3,-4(a5!) # 31fffffc <__l2_shared_end+0x15feeeb4>
1c008664:	18068363          	beqz	a3,1c0087ea <__kernel_rem_pio2f+0x376>
1c008668:	1c0017b7          	lui	a5,0x1c001
1c00866c:	0907a503          	lw	a0,144(a5) # 1c001090 <__clz_tab+0x3e0>
1c008670:	85a6                	mv	a1,s1
1c008672:	2169                	jal	1c008afc <scalbnf>
1c008674:	00241793          	slli	a5,s0,0x2
1c008678:	1010                	addi	a2,sp,32
1c00867a:	00f606b3          	add	a3,a2,a5
1c00867e:	1c001637          	lui	a2,0x1c001
1c008682:	02862e03          	lw	t3,40(a2) # 1c001028 <__clz_tab+0x378>
1c008686:	0a18                	addi	a4,sp,272
1c008688:	00f70833          	add	a6,a4,a5
1c00868c:	85a2                	mv	a1,s0
1c00868e:	1c05d363          	bgez	a1,1c008854 <__kernel_rem_pio2f+0x3e0>
1c008692:	0a10                	addi	a2,sp,272
1c008694:	00c786b3          	add	a3,a5,a2
1c008698:	4601                	li	a2,0
1c00869a:	40c405b3          	sub	a1,s0,a2
1c00869e:	1e05c863          	bltz	a1,1c00888e <__kernel_rem_pio2f+0x41a>
1c0086a2:	1c001737          	lui	a4,0x1c001
1c0086a6:	c0070713          	addi	a4,a4,-1024 # 1c000c00 <init_jk>
1c0086aa:	00261513          	slli	a0,a2,0x2
1c0086ae:	00c70e93          	addi	t4,a4,12
1c0086b2:	40a68e33          	sub	t3,a3,a0
1c0086b6:	00000813          	li	a6,0
1c0086ba:	4581                	li	a1,0
1c0086bc:	aa7d                	j	1c00887a <__kernel_rem_pio2f+0x406>
1c0086be:	1c001637          	lui	a2,0x1c001
1c0086c2:	02860613          	addi	a2,a2,40 # 1c001028 <__clz_tab+0x378>
1c0086c6:	4210                	lw	a2,0(a2)
1c0086c8:	15fd                	addi	a1,a1,-1
1c0086ca:	10c57753          	fmul.s	a4,a0,a2
1c0086ce:	1c001637          	lui	a2,0x1c001
1c0086d2:	0cc60613          	addi	a2,a2,204 # 1c0010cc <__clz_tab+0x41c>
1c0086d6:	4210                	lw	a2,0(a2)
1c0086d8:	c0071753          	fcvt.w.s	a4,a4,rtz
1c0086dc:	d0077753          	fcvt.s.w	a4,a4
1c0086e0:	50c7754b          	fnmsub.s	a0,a4,a2,a0
1c0086e4:	c0051553          	fcvt.w.s	a0,a0,rtz
1c0086e8:	00afa22b          	p.sw	a0,4(t6!)
1c0086ec:	00032503          	lw	a0,0(t1)
1c0086f0:	00a77553          	fadd.s	a0,a4,a0
1c0086f4:	b559                	j	1c00857a <__kernel_rem_pio2f+0x106>
1c0086f6:	e899                	bnez	s1,1c00870c <__kernel_rem_pio2f+0x298>
1c0086f8:	fffd8713          	addi	a4,s11,-1
1c0086fc:	070a                	slli	a4,a4,0x2
1c0086fe:	1290                	addi	a2,sp,352
1c008700:	9732                	add	a4,a4,a2
1c008702:	ec072703          	lw	a4,-320(a4)
1c008706:	40875d13          	srai	s10,a4,0x8
1c00870a:	b5cd                	j	1c0085ec <__kernel_rem_pio2f+0x178>
1c00870c:	1c001737          	lui	a4,0x1c001
1c008710:	08872703          	lw	a4,136(a4) # 1c001088 <__clz_tab+0x3d8>
1c008714:	4d01                	li	s10,0
1c008716:	a0870753          	fle.s	a4,a4,s0
1c00871a:	df01                	beqz	a4,1c008632 <__kernel_rem_pio2f+0x1be>
1c00871c:	4d09                	li	s10,2
1c00871e:	bdc9                	j	1c0085f0 <__kernel_rem_pio2f+0x17c>
1c008720:	000f2583          	lw	a1,0(t5)
1c008724:	00039c63          	bnez	t2,1c00873c <__kernel_rem_pio2f+0x2c8>
1c008728:	c591                	beqz	a1,1c008734 <__kernel_rem_pio2f+0x2c0>
1c00872a:	40b285b3          	sub	a1,t0,a1
1c00872e:	00bf2023          	sw	a1,0(t5)
1c008732:	4585                	li	a1,1
1c008734:	0505                	addi	a0,a0,1
1c008736:	0f11                	addi	t5,t5,4
1c008738:	83ae                	mv	t2,a1
1c00873a:	b5d1                	j	1c0085fe <__kernel_rem_pio2f+0x18a>
1c00873c:	40bf85b3          	sub	a1,t6,a1
1c008740:	00bf2023          	sw	a1,0(t5)
1c008744:	859e                	mv	a1,t2
1c008746:	b7fd                	j	1c008734 <__kernel_rem_pio2f+0x2c0>
1c008748:	fffd8593          	addi	a1,s11,-1
1c00874c:	058a                	slli	a1,a1,0x2
1c00874e:	1298                	addi	a4,sp,352
1c008750:	95ba                	add	a1,a1,a4
1c008752:	ec05a503          	lw	a0,-320(a1)
1c008756:	f0753533          	p.bclr	a0,a0,24,7
1c00875a:	eca5a023          	sw	a0,-320(a1)
1c00875e:	bd45                	j	1c00860e <__kernel_rem_pio2f+0x19a>
1c008760:	fffd8593          	addi	a1,s11,-1
1c008764:	058a                	slli	a1,a1,0x2
1c008766:	1290                	addi	a2,sp,352
1c008768:	95b2                	add	a1,a1,a2
1c00876a:	ec05a503          	lw	a0,-320(a1)
1c00876e:	f2653533          	p.bclr	a0,a0,25,6
1c008772:	b7e5                	j	1c00875a <__kernel_rem_pio2f+0x2e6>
1c008774:	000f2f83          	lw	t6,0(t5)
1c008778:	157d                	addi	a0,a0,-1
1c00877a:	01f5e5b3          	or	a1,a1,t6
1c00877e:	b5f9                	j	1c00864c <__kernel_rem_pio2f+0x1d8>
1c008780:	0585                	addi	a1,a1,1
1c008782:	ffc7250b          	p.lw	a0,-4(a4!)
1c008786:	dd6d                	beqz	a0,1c008780 <__kernel_rem_pio2f+0x30c>
1c008788:	0a18                	addi	a4,sp,272
1c00878a:	001d8f93          	addi	t6,s11,1
1c00878e:	004a0613          	addi	a2,s4,4
1c008792:	963a                	add	a2,a2,a4
1c008794:	01fb0f33          	add	t5,s6,t6
1c008798:	00dd8733          	add	a4,s11,a3
1c00879c:	0f0a                	slli	t5,t5,0x2
1c00879e:	070a                	slli	a4,a4,0x2
1c0087a0:	9f3e                	add	t5,t5,a5
1c0087a2:	9756                	add	a4,a4,s5
1c0087a4:	00bd8833          	add	a6,s11,a1
1c0087a8:	01f85763          	ble	t6,a6,1c0087b6 <__kernel_rem_pio2f+0x342>
1c0087ac:	8dc2                	mv	s11,a6
1c0087ae:	bb45                	j	1c00855e <__kernel_rem_pio2f+0xea>
1c0087b0:	4772                	lw	a4,28(sp)
1c0087b2:	4585                	li	a1,1
1c0087b4:	b7f9                	j	1c008782 <__kernel_rem_pio2f+0x30e>
1c0087b6:	004f250b          	p.lw	a0,4(t5!)
1c0087ba:	85ba                	mv	a1,a4
1c0087bc:	42e2                	lw	t0,24(sp)
1c0087be:	d0057553          	fcvt.s.w	a0,a0
1c0087c2:	4301                	li	t1,0
1c0087c4:	00a5a22b          	p.sw	a0,4(a1!)
1c0087c8:	00000513          	li	a0,0
1c0087cc:	006bd763          	ble	t1,s7,1c0087da <__kernel_rem_pio2f+0x366>
1c0087d0:	00a6222b          	p.sw	a0,4(a2!)
1c0087d4:	0f85                	addi	t6,t6,1
1c0087d6:	872e                	mv	a4,a1
1c0087d8:	bfc1                	j	1c0087a8 <__kernel_rem_pio2f+0x334>
1c0087da:	0042a40b          	p.lw	s0,4(t0!)
1c0087de:	ffc7238b          	p.lw	t2,-4(a4!)
1c0087e2:	0305                	addi	t1,t1,1
1c0087e4:	50747543          	fmadd.s	a0,s0,t2,a0
1c0087e8:	b7d5                	j	1c0087cc <__kernel_rem_pio2f+0x358>
1c0087ea:	147d                	addi	s0,s0,-1
1c0087ec:	14e1                	addi	s1,s1,-8
1c0087ee:	bd8d                	j	1c008660 <__kernel_rem_pio2f+0x1ec>
1c0087f0:	8522                	mv	a0,s0
1c0087f2:	409005b3          	neg	a1,s1
1c0087f6:	2619                	jal	1c008afc <scalbnf>
1c0087f8:	1c0017b7          	lui	a5,0x1c001
1c0087fc:	0cc7a683          	lw	a3,204(a5) # 1c0010cc <__clz_tab+0x41c>
1c008800:	a0a687d3          	fle.s	a5,a3,a0
1c008804:	cf9d                	beqz	a5,1c008842 <__kernel_rem_pio2f+0x3ce>
1c008806:	1c0017b7          	lui	a5,0x1c001
1c00880a:	0287a783          	lw	a5,40(a5) # 1c001028 <__clz_tab+0x378>
1c00880e:	1298                	addi	a4,sp,352
1c008810:	001d8413          	addi	s0,s11,1
1c008814:	10f577d3          	fmul.s	a5,a0,a5
1c008818:	01470633          	add	a2,a4,s4
1c00881c:	04a1                	addi	s1,s1,8
1c00881e:	c00797d3          	fcvt.w.s	a5,a5,rtz
1c008822:	d007f7d3          	fcvt.s.w	a5,a5
1c008826:	50d7f54b          	fnmsub.s	a0,a5,a3,a0
1c00882a:	c00797d3          	fcvt.w.s	a5,a5,rtz
1c00882e:	00241693          	slli	a3,s0,0x2
1c008832:	96ba                	add	a3,a3,a4
1c008834:	c0051553          	fcvt.w.s	a0,a0,rtz
1c008838:	eca62023          	sw	a0,-320(a2)
1c00883c:	ecf6a023          	sw	a5,-320(a3)
1c008840:	b525                	j	1c008668 <__kernel_rem_pio2f+0x1f4>
1c008842:	c0051553          	fcvt.w.s	a0,a0,rtz
1c008846:	129c                	addi	a5,sp,352
1c008848:	01478633          	add	a2,a5,s4
1c00884c:	eca62023          	sw	a0,-320(a2)
1c008850:	846e                	mv	s0,s11
1c008852:	bd19                	j	1c008668 <__kernel_rem_pio2f+0x1f4>
1c008854:	ffc6a60b          	p.lw	a2,-4(a3!)
1c008858:	15fd                	addi	a1,a1,-1
1c00885a:	d0067653          	fcvt.s.w	a2,a2
1c00885e:	10a67653          	fmul.s	a2,a2,a0
1c008862:	11c57553          	fmul.s	a0,a0,t3
1c008866:	fec82e2b          	p.sw	a2,-4(a6!)
1c00886a:	b515                	j	1c00868e <__kernel_rem_pio2f+0x21a>
1c00886c:	004eaf8b          	p.lw	t6,4(t4!)
1c008870:	004e2f0b          	p.lw	t5,4(t3!)
1c008874:	0585                	addi	a1,a1,1
1c008876:	81eff843          	fmadd.s	a6,t6,t5,a6
1c00887a:	00b94463          	blt	s2,a1,1c008882 <__kernel_rem_pio2f+0x40e>
1c00887e:	feb657e3          	ble	a1,a2,1c00886c <__kernel_rem_pio2f+0x3f8>
1c008882:	1298                	addi	a4,sp,352
1c008884:	953a                	add	a0,a0,a4
1c008886:	f7052023          	sw	a6,-160(a0) # feffff60 <pulp__FC+0xfeffff61>
1c00888a:	0605                	addi	a2,a2,1
1c00888c:	b539                	j	1c00869a <__kernel_rem_pio2f+0x226>
1c00888e:	4689                	li	a3,2
1c008890:	0536c463          	blt	a3,s3,1c0088d8 <__kernel_rem_pio2f+0x464>
1c008894:	09304163          	bgtz	s3,1c008916 <__kernel_rem_pio2f+0x4a2>
1c008898:	0a098c63          	beqz	s3,1c008950 <__kernel_rem_pio2f+0x4dc>
1c00889c:	19c12083          	lw	ra,412(sp)
1c0088a0:	19812403          	lw	s0,408(sp)
1c0088a4:	f83c3533          	p.bclr	a0,s8,28,3
1c0088a8:	19412483          	lw	s1,404(sp)
1c0088ac:	19012903          	lw	s2,400(sp)
1c0088b0:	18c12983          	lw	s3,396(sp)
1c0088b4:	18812a03          	lw	s4,392(sp)
1c0088b8:	18412a83          	lw	s5,388(sp)
1c0088bc:	18012b03          	lw	s6,384(sp)
1c0088c0:	17c12b83          	lw	s7,380(sp)
1c0088c4:	17812c03          	lw	s8,376(sp)
1c0088c8:	17412c83          	lw	s9,372(sp)
1c0088cc:	17012d03          	lw	s10,368(sp)
1c0088d0:	16c12d83          	lw	s11,364(sp)
1c0088d4:	611d                	addi	sp,sp,416
1c0088d6:	8082                	ret
1c0088d8:	fc39b2e3          	p.bneimm	s3,3,1c00889c <__kernel_rem_pio2f+0x428>
1c0088dc:	0190                	addi	a2,sp,192
1c0088de:	97b2                	add	a5,a5,a2
1c0088e0:	86be                	mv	a3,a5
1c0088e2:	85a2                	mv	a1,s0
1c0088e4:	16f1                	addi	a3,a3,-4
1c0088e6:	0ab04363          	bgtz	a1,1c00898c <__kernel_rem_pio2f+0x518>
1c0088ea:	86be                	mv	a3,a5
1c0088ec:	85a2                	mv	a1,s0
1c0088ee:	4e05                	li	t3,1
1c0088f0:	16f1                	addi	a3,a3,-4
1c0088f2:	0abe4a63          	blt	t3,a1,1c0089a6 <__kernel_rem_pio2f+0x532>
1c0088f6:	00000693          	li	a3,0
1c0088fa:	4605                	li	a2,1
1c0088fc:	0c864263          	blt	a2,s0,1c0089c0 <__kernel_rem_pio2f+0x54c>
1c008900:	460e                	lw	a2,192(sp)
1c008902:	479e                	lw	a5,196(sp)
1c008904:	0c0d1463          	bnez	s10,1c0089cc <__kernel_rem_pio2f+0x558>
1c008908:	00cca023          	sw	a2,0(s9)
1c00890c:	00fca223          	sw	a5,4(s9)
1c008910:	00dca423          	sw	a3,8(s9)
1c008914:	b761                	j	1c00889c <__kernel_rem_pio2f+0x428>
1c008916:	00000693          	li	a3,0
1c00891a:	0198                	addi	a4,sp,192
1c00891c:	97ba                	add	a5,a5,a4
1c00891e:	8622                	mv	a2,s0
1c008920:	04065b63          	bgez	a2,1c008976 <__kernel_rem_pio2f+0x502>
1c008924:	87b6                	mv	a5,a3
1c008926:	000d0463          	beqz	s10,1c00892e <__kernel_rem_pio2f+0x4ba>
1c00892a:	20d697d3          	fneg.s	a5,a3a3
1c00892e:	00fca023          	sw	a5,0(s9)
1c008932:	478e                	lw	a5,192(sp)
1c008934:	4605                	li	a2,1
1c008936:	08d7f7d3          	fsub.s	a5,a5,a3
1c00893a:	0194                	addi	a3,sp,192
1c00893c:	0691                	addi	a3,a3,4
1c00893e:	04c45263          	ble	a2,s0,1c008982 <__kernel_rem_pio2f+0x50e>
1c008942:	000d0463          	beqz	s10,1c00894a <__kernel_rem_pio2f+0x4d6>
1c008946:	20f797d3          	fneg.s	a5,a5a5
1c00894a:	00fca223          	sw	a5,4(s9)
1c00894e:	b7b9                	j	1c00889c <__kernel_rem_pio2f+0x428>
1c008950:	00000693          	li	a3,0
1c008954:	0190                	addi	a2,sp,192
1c008956:	97b2                	add	a5,a5,a2
1c008958:	00045963          	bgez	s0,1c00896a <__kernel_rem_pio2f+0x4f6>
1c00895c:	000d0463          	beqz	s10,1c008964 <__kernel_rem_pio2f+0x4f0>
1c008960:	20d696d3          	fneg.s	a3,a3a3
1c008964:	00dca023          	sw	a3,0(s9)
1c008968:	bf15                	j	1c00889c <__kernel_rem_pio2f+0x428>
1c00896a:	ffc7a60b          	p.lw	a2,-4(a5!)
1c00896e:	147d                	addi	s0,s0,-1
1c008970:	00c6f6d3          	fadd.s	a3,a3,a2
1c008974:	b7d5                	j	1c008958 <__kernel_rem_pio2f+0x4e4>
1c008976:	ffc7a58b          	p.lw	a1,-4(a5!)
1c00897a:	167d                	addi	a2,a2,-1
1c00897c:	00b6f6d3          	fadd.s	a3,a3,a1
1c008980:	b745                	j	1c008920 <__kernel_rem_pio2f+0x4ac>
1c008982:	428c                	lw	a1,0(a3)
1c008984:	0605                	addi	a2,a2,1
1c008986:	00b7f7d3          	fadd.s	a5,a5,a1
1c00898a:	bf4d                	j	1c00893c <__kernel_rem_pio2f+0x4c8>
1c00898c:	4290                	lw	a2,0(a3)
1c00898e:	0046a803          	lw	a6,4(a3)
1c008992:	15fd                	addi	a1,a1,-1
1c008994:	01067553          	fadd.s	a0,a2,a6
1c008998:	08a67653          	fsub.s	a2,a2,a0
1c00899c:	c288                	sw	a0,0(a3)
1c00899e:	01067653          	fadd.s	a2,a2,a6
1c0089a2:	c2d0                	sw	a2,4(a3)
1c0089a4:	b781                	j	1c0088e4 <__kernel_rem_pio2f+0x470>
1c0089a6:	4290                	lw	a2,0(a3)
1c0089a8:	0046a803          	lw	a6,4(a3)
1c0089ac:	15fd                	addi	a1,a1,-1
1c0089ae:	01067553          	fadd.s	a0,a2,a6
1c0089b2:	08a67653          	fsub.s	a2,a2,a0
1c0089b6:	c288                	sw	a0,0(a3)
1c0089b8:	01067653          	fadd.s	a2,a2,a6
1c0089bc:	c2d0                	sw	a2,4(a3)
1c0089be:	bf0d                	j	1c0088f0 <__kernel_rem_pio2f+0x47c>
1c0089c0:	ffc7a58b          	p.lw	a1,-4(a5!)
1c0089c4:	147d                	addi	s0,s0,-1
1c0089c6:	00b6f6d3          	fadd.s	a3,a3,a1
1c0089ca:	bf0d                	j	1c0088fc <__kernel_rem_pio2f+0x488>
1c0089cc:	20c61653          	fneg.s	a2,a2a2
1c0089d0:	20f797d3          	fneg.s	a5,a5a5
1c0089d4:	20d696d3          	fneg.s	a3,a3a3
1c0089d8:	00cca023          	sw	a2,0(s9)
1c0089dc:	00fca223          	sw	a5,4(s9)
1c0089e0:	bf05                	j	1c008910 <__kernel_rem_pio2f+0x49c>

1c0089e2 <__kernel_sinf>:
1c0089e2:	c1f53733          	p.bclr	a4,a0,0,31
1c0089e6:	320007b7          	lui	a5,0x32000
1c0089ea:	00f75563          	ble	a5,a4,1c0089f4 <__kernel_sinf+0x12>
1c0089ee:	c00517d3          	fcvt.w.s	a5,a0,rtz
1c0089f2:	cfad                	beqz	a5,1c008a6c <__kernel_sinf+0x8a>
1c0089f4:	10a57753          	fmul.s	a4,a0,a0
1c0089f8:	1c0017b7          	lui	a5,0x1c001
1c0089fc:	1c001837          	lui	a6,0x1c001
1c008a00:	0f482803          	lw	a6,244(a6) # 1c0010f4 <__clz_tab+0x444>
1c008a04:	0f07a783          	lw	a5,240(a5) # 1c0010f0 <__clz_tab+0x440>
1c008a08:	10e576d3          	fmul.s	a3,a0,a4
1c008a0c:	80f777c3          	fmadd.s	a5,a4,a5,a6
1c008a10:	1c001837          	lui	a6,0x1c001
1c008a14:	0f882803          	lw	a6,248(a6) # 1c0010f8 <__clz_tab+0x448>
1c008a18:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a1c:	1c001837          	lui	a6,0x1c001
1c008a20:	0fc82803          	lw	a6,252(a6) # 1c0010fc <__clz_tab+0x44c>
1c008a24:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a28:	1c001837          	lui	a6,0x1c001
1c008a2c:	10082803          	lw	a6,256(a6) # 1c001100 <__clz_tab+0x450>
1c008a30:	80e7f7c3          	fmadd.s	a5,a5,a4,a6
1c008a34:	ea11                	bnez	a2,1c008a48 <__kernel_sinf+0x66>
1c008a36:	1c001637          	lui	a2,0x1c001
1c008a3a:	10462583          	lw	a1,260(a2) # 1c001104 <__clz_tab+0x454>
1c008a3e:	58f777c3          	fmadd.s	a5,a4,a5,a1
1c008a42:	50d7f543          	fmadd.s	a0,a5,a3,a0
1c008a46:	8082                	ret
1c008a48:	10f6f7d3          	fmul.s	a5,a3,a5
1c008a4c:	1c001637          	lui	a2,0x1c001
1c008a50:	08862603          	lw	a2,136(a2) # 1c001088 <__clz_tab+0x3d8>
1c008a54:	78c5f7c7          	fmsub.s	a5,a1,a2,a5
1c008a58:	58e7f747          	fmsub.s	a4,a5,a4,a1
1c008a5c:	1c0017b7          	lui	a5,0x1c001
1c008a60:	1087a783          	lw	a5,264(a5) # 1c001108 <__clz_tab+0x458>
1c008a64:	70f6f6c3          	fmadd.s	a3,a3,a5,a4
1c008a68:	08d57553          	fsub.s	a0,a0,a3
1c008a6c:	8082                	ret

1c008a6e <fabsf>:
1c008a6e:	c1f53533          	p.bclr	a0,a0,0,31
1c008a72:	8082                	ret

1c008a74 <floorf>:
1c008a74:	c1f53733          	p.bclr	a4,a0,0,31
1c008a78:	01775693          	srli	a3,a4,0x17
1c008a7c:	f8168693          	addi	a3,a3,-127
1c008a80:	47d9                	li	a5,22
1c008a82:	06d7c463          	blt	a5,a3,1c008aea <floorf+0x76>
1c008a86:	87aa                	mv	a5,a0
1c008a88:	0206d463          	bgez	a3,1c008ab0 <floorf+0x3c>
1c008a8c:	1c0016b7          	lui	a3,0x1c001
1c008a90:	10c6a683          	lw	a3,268(a3) # 1c00110c <__clz_tab+0x45c>
1c008a94:	00d57553          	fadd.s	a0,a0,a3
1c008a98:	00000693          	li	a3,0
1c008a9c:	a0a69553          	flt.s	a0,a3,a0
1c008aa0:	c511                	beqz	a0,1c008aac <floorf+0x38>
1c008aa2:	0407db63          	bgez	a5,1c008af8 <floorf+0x84>
1c008aa6:	c319                	beqz	a4,1c008aac <floorf+0x38>
1c008aa8:	bf8007b7          	lui	a5,0xbf800
1c008aac:	853e                	mv	a0,a5
1c008aae:	8082                	ret
1c008ab0:	00800637          	lui	a2,0x800
1c008ab4:	fff60713          	addi	a4,a2,-1 # 7fffff <__l1_heap_size+0x700983>
1c008ab8:	40d75733          	sra	a4,a4,a3
1c008abc:	00a775b3          	and	a1,a4,a0
1c008ac0:	d5fd                	beqz	a1,1c008aae <floorf+0x3a>
1c008ac2:	1c0015b7          	lui	a1,0x1c001
1c008ac6:	10c5a583          	lw	a1,268(a1) # 1c00110c <__clz_tab+0x45c>
1c008aca:	00b57553          	fadd.s	a0,a0,a1
1c008ace:	00000593          	li	a1,0
1c008ad2:	a0a59553          	flt.s	a0,a1,a0
1c008ad6:	d979                	beqz	a0,1c008aac <floorf+0x38>
1c008ad8:	0007d563          	bgez	a5,1c008ae2 <floorf+0x6e>
1c008adc:	40d656b3          	sra	a3,a2,a3
1c008ae0:	97b6                	add	a5,a5,a3
1c008ae2:	fff74713          	not	a4,a4
1c008ae6:	8ff9                	and	a5,a5,a4
1c008ae8:	b7d1                	j	1c008aac <floorf+0x38>
1c008aea:	7f8007b7          	lui	a5,0x7f800
1c008aee:	fcf760e3          	bltu	a4,a5,1c008aae <floorf+0x3a>
1c008af2:	00a57553          	fadd.s	a0,a0,a0
1c008af6:	8082                	ret
1c008af8:	4781                	li	a5,0
1c008afa:	bf4d                	j	1c008aac <floorf+0x38>

1c008afc <scalbnf>:
1c008afc:	c1f537b3          	p.bclr	a5,a0,0,31
1c008b00:	cfcd                	beqz	a5,1c008bba <scalbnf+0xbe>
1c008b02:	7f800737          	lui	a4,0x7f800
1c008b06:	00e7e563          	bltu	a5,a4,1c008b10 <scalbnf+0x14>
1c008b0a:	00a57553          	fadd.s	a0,a0,a0
1c008b0e:	8082                	ret
1c008b10:	00800737          	lui	a4,0x800
1c008b14:	04e7fc63          	bleu	a4,a5,1c008b6c <scalbnf+0x70>
1c008b18:	1c0017b7          	lui	a5,0x1c001
1c008b1c:	1107a783          	lw	a5,272(a5) # 1c001110 <__clz_tab+0x460>
1c008b20:	10f57553          	fmul.s	a0,a0,a5
1c008b24:	77d1                	lui	a5,0xffff4
1c008b26:	cb078793          	addi	a5,a5,-848 # ffff3cb0 <pulp__FC+0xffff3cb1>
1c008b2a:	02f5ca63          	blt	a1,a5,1c008b5e <scalbnf+0x62>
1c008b2e:	41755793          	srai	a5,a0,0x17
1c008b32:	ee87b7b3          	p.bclr	a5,a5,23,8
1c008b36:	872a                	mv	a4,a0
1c008b38:	179d                	addi	a5,a5,-25
1c008b3a:	1141                	addi	sp,sp,-16
1c008b3c:	c606                	sw	ra,12(sp)
1c008b3e:	c422                	sw	s0,8(sp)
1c008b40:	97ae                	add	a5,a5,a1
1c008b42:	0fe00693          	li	a3,254
1c008b46:	02f6d663          	ble	a5,a3,1c008b72 <scalbnf+0x76>
1c008b4a:	1c0017b7          	lui	a5,0x1c001
1c008b4e:	10c7a403          	lw	s0,268(a5) # 1c00110c <__clz_tab+0x45c>
1c008b52:	85aa                	mv	a1,a0
1c008b54:	8522                	mv	a0,s0
1c008b56:	209d                	jal	1c008bbc <copysignf>
1c008b58:	10857553          	fmul.s	a0,a0,s0
1c008b5c:	a00d                	j	1c008b7e <scalbnf+0x82>
1c008b5e:	1c0017b7          	lui	a5,0x1c001
1c008b62:	1147a783          	lw	a5,276(a5) # 1c001114 <__clz_tab+0x464>
1c008b66:	10f57553          	fmul.s	a0,a0,a5
1c008b6a:	8082                	ret
1c008b6c:	872a                	mv	a4,a0
1c008b6e:	83dd                	srli	a5,a5,0x17
1c008b70:	b7e9                	j	1c008b3a <scalbnf+0x3e>
1c008b72:	00f05a63          	blez	a5,1c008b86 <scalbnf+0x8a>
1c008b76:	cf773533          	p.bclr	a0,a4,7,23
1c008b7a:	07de                	slli	a5,a5,0x17
1c008b7c:	8d5d                	or	a0,a0,a5
1c008b7e:	40b2                	lw	ra,12(sp)
1c008b80:	4422                	lw	s0,8(sp)
1c008b82:	0141                	addi	sp,sp,16
1c008b84:	8082                	ret
1c008b86:	56a9                	li	a3,-22
1c008b88:	00d7dc63          	ble	a3,a5,1c008ba0 <scalbnf+0xa4>
1c008b8c:	67b1                	lui	a5,0xc
1c008b8e:	35078793          	addi	a5,a5,848 # c350 <_l1_preload_size+0xb9d0>
1c008b92:	fab7cce3          	blt	a5,a1,1c008b4a <scalbnf+0x4e>
1c008b96:	1c0017b7          	lui	a5,0x1c001
1c008b9a:	1147a403          	lw	s0,276(a5) # 1c001114 <__clz_tab+0x464>
1c008b9e:	bf55                	j	1c008b52 <scalbnf+0x56>
1c008ba0:	01978513          	addi	a0,a5,25
1c008ba4:	1c0017b7          	lui	a5,0x1c001
1c008ba8:	1187a783          	lw	a5,280(a5) # 1c001118 <__clz_tab+0x468>
1c008bac:	055e                	slli	a0,a0,0x17
1c008bae:	cf773733          	p.bclr	a4,a4,7,23
1c008bb2:	8d59                	or	a0,a0,a4
1c008bb4:	10f57553          	fmul.s	a0,a0,a5
1c008bb8:	b7d9                	j	1c008b7e <scalbnf+0x82>
1c008bba:	8082                	ret

1c008bbc <copysignf>:
1c008bbc:	fc0525b3          	p.insert	a1,a0,30,0
1c008bc0:	852e                	mv	a0,a1
1c008bc2:	8082                	ret

1c008bc4 <__udivdi3>:
1c008bc4:	87b2                	mv	a5,a2
1c008bc6:	8736                	mv	a4,a3
1c008bc8:	88aa                	mv	a7,a0
1c008bca:	882e                	mv	a6,a1
1c008bcc:	1e069d63          	bnez	a3,1c008dc6 <__udivdi3+0x202>
1c008bd0:	1c001337          	lui	t1,0x1c001
1c008bd4:	cb030313          	addi	t1,t1,-848 # 1c000cb0 <__clz_tab>
1c008bd8:	0ac5fd63          	bleu	a2,a1,1c008c92 <__udivdi3+0xce>
1c008bdc:	6741                	lui	a4,0x10
1c008bde:	0ae67363          	bleu	a4,a2,1c008c84 <__udivdi3+0xc0>
1c008be2:	0ff00693          	li	a3,255
1c008be6:	00c6b6b3          	sltu	a3,a3,a2
1c008bea:	068e                	slli	a3,a3,0x3
1c008bec:	00d65733          	srl	a4,a2,a3
1c008bf0:	933a                	add	t1,t1,a4
1c008bf2:	00034703          	lbu	a4,0(t1)
1c008bf6:	02000313          	li	t1,32
1c008bfa:	96ba                	add	a3,a3,a4
1c008bfc:	40d30333          	sub	t1,t1,a3
1c008c00:	00030c63          	beqz	t1,1c008c18 <__udivdi3+0x54>
1c008c04:	00659733          	sll	a4,a1,t1
1c008c08:	00d556b3          	srl	a3,a0,a3
1c008c0c:	006617b3          	sll	a5,a2,t1
1c008c10:	00e6e833          	or	a6,a3,a4
1c008c14:	006518b3          	sll	a7,a0,t1
1c008c18:	0107d513          	srli	a0,a5,0x10
1c008c1c:	02a85733          	divu	a4,a6,a0
1c008c20:	1007d5b3          	p.exthz	a1,a5
1c008c24:	0108d693          	srli	a3,a7,0x10
1c008c28:	02a87633          	remu	a2,a6,a0
1c008c2c:	02e58833          	mul	a6,a1,a4
1c008c30:	0642                	slli	a2,a2,0x10
1c008c32:	8ed1                	or	a3,a3,a2
1c008c34:	863a                	mv	a2,a4
1c008c36:	0106fc63          	bleu	a6,a3,1c008c4e <__udivdi3+0x8a>
1c008c3a:	96be                	add	a3,a3,a5
1c008c3c:	fff70613          	addi	a2,a4,-1 # ffff <_l1_preload_size+0xf67f>
1c008c40:	00f6e763          	bltu	a3,a5,1c008c4e <__udivdi3+0x8a>
1c008c44:	0106f563          	bleu	a6,a3,1c008c4e <__udivdi3+0x8a>
1c008c48:	ffe70613          	addi	a2,a4,-2
1c008c4c:	96be                	add	a3,a3,a5
1c008c4e:	410686b3          	sub	a3,a3,a6
1c008c52:	02a6f833          	remu	a6,a3,a0
1c008c56:	02a6d6b3          	divu	a3,a3,a0
1c008c5a:	df0828b3          	p.insert	a7,a6,15,16
1c008c5e:	02d58733          	mul	a4,a1,a3
1c008c62:	8536                	mv	a0,a3
1c008c64:	00e8fb63          	bleu	a4,a7,1c008c7a <__udivdi3+0xb6>
1c008c68:	98be                	add	a7,a7,a5
1c008c6a:	fff68513          	addi	a0,a3,-1
1c008c6e:	00f8e663          	bltu	a7,a5,1c008c7a <__udivdi3+0xb6>
1c008c72:	00e8f463          	bleu	a4,a7,1c008c7a <__udivdi3+0xb6>
1c008c76:	ffe68513          	addi	a0,a3,-2
1c008c7a:	01061793          	slli	a5,a2,0x10
1c008c7e:	8fc9                	or	a5,a5,a0
1c008c80:	4801                	li	a6,0
1c008c82:	a06d                	j	1c008d2c <__udivdi3+0x168>
1c008c84:	01000737          	lui	a4,0x1000
1c008c88:	46c1                	li	a3,16
1c008c8a:	f6e661e3          	bltu	a2,a4,1c008bec <__udivdi3+0x28>
1c008c8e:	46e1                	li	a3,24
1c008c90:	bfb1                	j	1c008bec <__udivdi3+0x28>
1c008c92:	e601                	bnez	a2,1c008c9a <__udivdi3+0xd6>
1c008c94:	4685                	li	a3,1
1c008c96:	02c6d7b3          	divu	a5,a3,a2
1c008c9a:	66c1                	lui	a3,0x10
1c008c9c:	08d7fb63          	bleu	a3,a5,1c008d32 <__udivdi3+0x16e>
1c008ca0:	0ff00693          	li	a3,255
1c008ca4:	00f6f363          	bleu	a5,a3,1c008caa <__udivdi3+0xe6>
1c008ca8:	4721                	li	a4,8
1c008caa:	00e7d6b3          	srl	a3,a5,a4
1c008cae:	9336                	add	t1,t1,a3
1c008cb0:	00034683          	lbu	a3,0(t1)
1c008cb4:	02000613          	li	a2,32
1c008cb8:	96ba                	add	a3,a3,a4
1c008cba:	8e15                	sub	a2,a2,a3
1c008cbc:	e251                	bnez	a2,1c008d40 <__udivdi3+0x17c>
1c008cbe:	40f58733          	sub	a4,a1,a5
1c008cc2:	4805                	li	a6,1
1c008cc4:	0107d513          	srli	a0,a5,0x10
1c008cc8:	1007d5b3          	p.exthz	a1,a5
1c008ccc:	0108d693          	srli	a3,a7,0x10
1c008cd0:	02a77633          	remu	a2,a4,a0
1c008cd4:	02a75733          	divu	a4,a4,a0
1c008cd8:	0642                	slli	a2,a2,0x10
1c008cda:	8ed1                	or	a3,a3,a2
1c008cdc:	02e58333          	mul	t1,a1,a4
1c008ce0:	863a                	mv	a2,a4
1c008ce2:	0066fc63          	bleu	t1,a3,1c008cfa <__udivdi3+0x136>
1c008ce6:	96be                	add	a3,a3,a5
1c008ce8:	fff70613          	addi	a2,a4,-1 # ffffff <__l1_heap_size+0xf00983>
1c008cec:	00f6e763          	bltu	a3,a5,1c008cfa <__udivdi3+0x136>
1c008cf0:	0066f563          	bleu	t1,a3,1c008cfa <__udivdi3+0x136>
1c008cf4:	ffe70613          	addi	a2,a4,-2
1c008cf8:	96be                	add	a3,a3,a5
1c008cfa:	406686b3          	sub	a3,a3,t1
1c008cfe:	02a6f333          	remu	t1,a3,a0
1c008d02:	02a6d6b3          	divu	a3,a3,a0
1c008d06:	df0328b3          	p.insert	a7,t1,15,16
1c008d0a:	02d58733          	mul	a4,a1,a3
1c008d0e:	8536                	mv	a0,a3
1c008d10:	00e8fb63          	bleu	a4,a7,1c008d26 <__udivdi3+0x162>
1c008d14:	98be                	add	a7,a7,a5
1c008d16:	fff68513          	addi	a0,a3,-1 # ffff <_l1_preload_size+0xf67f>
1c008d1a:	00f8e663          	bltu	a7,a5,1c008d26 <__udivdi3+0x162>
1c008d1e:	00e8f463          	bleu	a4,a7,1c008d26 <__udivdi3+0x162>
1c008d22:	ffe68513          	addi	a0,a3,-2
1c008d26:	01061793          	slli	a5,a2,0x10
1c008d2a:	8fc9                	or	a5,a5,a0
1c008d2c:	853e                	mv	a0,a5
1c008d2e:	85c2                	mv	a1,a6
1c008d30:	8082                	ret
1c008d32:	010006b7          	lui	a3,0x1000
1c008d36:	4741                	li	a4,16
1c008d38:	f6d7e9e3          	bltu	a5,a3,1c008caa <__udivdi3+0xe6>
1c008d3c:	4761                	li	a4,24
1c008d3e:	b7b5                	j	1c008caa <__udivdi3+0xe6>
1c008d40:	00c797b3          	sll	a5,a5,a2
1c008d44:	00d5d333          	srl	t1,a1,a3
1c008d48:	0107de13          	srli	t3,a5,0x10
1c008d4c:	00c59733          	sll	a4,a1,a2
1c008d50:	00c518b3          	sll	a7,a0,a2
1c008d54:	00d555b3          	srl	a1,a0,a3
1c008d58:	03c35533          	divu	a0,t1,t3
1c008d5c:	8dd9                	or	a1,a1,a4
1c008d5e:	1007d733          	p.exthz	a4,a5
1c008d62:	0105d693          	srli	a3,a1,0x10
1c008d66:	03c37633          	remu	a2,t1,t3
1c008d6a:	882a                	mv	a6,a0
1c008d6c:	02a70333          	mul	t1,a4,a0
1c008d70:	0642                	slli	a2,a2,0x10
1c008d72:	8ed1                	or	a3,a3,a2
1c008d74:	0066fc63          	bleu	t1,a3,1c008d8c <__udivdi3+0x1c8>
1c008d78:	96be                	add	a3,a3,a5
1c008d7a:	fff50813          	addi	a6,a0,-1
1c008d7e:	00f6e763          	bltu	a3,a5,1c008d8c <__udivdi3+0x1c8>
1c008d82:	0066f563          	bleu	t1,a3,1c008d8c <__udivdi3+0x1c8>
1c008d86:	ffe50813          	addi	a6,a0,-2
1c008d8a:	96be                	add	a3,a3,a5
1c008d8c:	406686b3          	sub	a3,a3,t1
1c008d90:	03c6f633          	remu	a2,a3,t3
1c008d94:	03c6d6b3          	divu	a3,a3,t3
1c008d98:	df0625b3          	p.insert	a1,a2,15,16
1c008d9c:	02d70733          	mul	a4,a4,a3
1c008da0:	8636                	mv	a2,a3
1c008da2:	00e5fc63          	bleu	a4,a1,1c008dba <__udivdi3+0x1f6>
1c008da6:	95be                	add	a1,a1,a5
1c008da8:	fff68613          	addi	a2,a3,-1 # ffffff <__l1_heap_size+0xf00983>
1c008dac:	00f5e763          	bltu	a1,a5,1c008dba <__udivdi3+0x1f6>
1c008db0:	00e5f563          	bleu	a4,a1,1c008dba <__udivdi3+0x1f6>
1c008db4:	ffe68613          	addi	a2,a3,-2
1c008db8:	95be                	add	a1,a1,a5
1c008dba:	0842                	slli	a6,a6,0x10
1c008dbc:	40e58733          	sub	a4,a1,a4
1c008dc0:	00c86833          	or	a6,a6,a2
1c008dc4:	b701                	j	1c008cc4 <__udivdi3+0x100>
1c008dc6:	12d5ea63          	bltu	a1,a3,1c008efa <__udivdi3+0x336>
1c008dca:	67c1                	lui	a5,0x10
1c008dcc:	02f6fd63          	bleu	a5,a3,1c008e06 <__udivdi3+0x242>
1c008dd0:	0ff00793          	li	a5,255
1c008dd4:	00d7b8b3          	sltu	a7,a5,a3
1c008dd8:	088e                	slli	a7,a7,0x3
1c008dda:	1c001737          	lui	a4,0x1c001
1c008dde:	0116d7b3          	srl	a5,a3,a7
1c008de2:	cb070713          	addi	a4,a4,-848 # 1c000cb0 <__clz_tab>
1c008de6:	97ba                	add	a5,a5,a4
1c008de8:	0007c783          	lbu	a5,0(a5) # 10000 <_l1_preload_size+0xf680>
1c008dec:	02000813          	li	a6,32
1c008df0:	97c6                	add	a5,a5,a7
1c008df2:	40f80833          	sub	a6,a6,a5
1c008df6:	00081f63          	bnez	a6,1c008e14 <__udivdi3+0x250>
1c008dfa:	4785                	li	a5,1
1c008dfc:	f2b6e8e3          	bltu	a3,a1,1c008d2c <__udivdi3+0x168>
1c008e00:	04a637b3          	p.sletu	a5,a2,a0
1c008e04:	b725                	j	1c008d2c <__udivdi3+0x168>
1c008e06:	010007b7          	lui	a5,0x1000
1c008e0a:	48c1                	li	a7,16
1c008e0c:	fcf6e7e3          	bltu	a3,a5,1c008dda <__udivdi3+0x216>
1c008e10:	48e1                	li	a7,24
1c008e12:	b7e1                	j	1c008dda <__udivdi3+0x216>
1c008e14:	00f658b3          	srl	a7,a2,a5
1c008e18:	010696b3          	sll	a3,a3,a6
1c008e1c:	00d8e6b3          	or	a3,a7,a3
1c008e20:	00f5d333          	srl	t1,a1,a5
1c008e24:	0106de13          	srli	t3,a3,0x10
1c008e28:	00f55733          	srl	a4,a0,a5
1c008e2c:	03c377b3          	remu	a5,t1,t3
1c008e30:	1006d8b3          	p.exthz	a7,a3
1c008e34:	010595b3          	sll	a1,a1,a6
1c008e38:	8f4d                	or	a4,a4,a1
1c008e3a:	01075593          	srli	a1,a4,0x10
1c008e3e:	01061633          	sll	a2,a2,a6
1c008e42:	03c35333          	divu	t1,t1,t3
1c008e46:	07c2                	slli	a5,a5,0x10
1c008e48:	8ddd                	or	a1,a1,a5
1c008e4a:	02688eb3          	mul	t4,a7,t1
1c008e4e:	879a                	mv	a5,t1
1c008e50:	01d5fc63          	bleu	t4,a1,1c008e68 <__udivdi3+0x2a4>
1c008e54:	95b6                	add	a1,a1,a3
1c008e56:	fff30793          	addi	a5,t1,-1
1c008e5a:	00d5e763          	bltu	a1,a3,1c008e68 <__udivdi3+0x2a4>
1c008e5e:	01d5f563          	bleu	t4,a1,1c008e68 <__udivdi3+0x2a4>
1c008e62:	ffe30793          	addi	a5,t1,-2
1c008e66:	95b6                	add	a1,a1,a3
1c008e68:	41d585b3          	sub	a1,a1,t4
1c008e6c:	03c5f333          	remu	t1,a1,t3
1c008e70:	03c5d5b3          	divu	a1,a1,t3
1c008e74:	df032733          	p.insert	a4,t1,15,16
1c008e78:	02b888b3          	mul	a7,a7,a1
1c008e7c:	832e                	mv	t1,a1
1c008e7e:	01177c63          	bleu	a7,a4,1c008e96 <__udivdi3+0x2d2>
1c008e82:	9736                	add	a4,a4,a3
1c008e84:	fff58313          	addi	t1,a1,-1
1c008e88:	00d76763          	bltu	a4,a3,1c008e96 <__udivdi3+0x2d2>
1c008e8c:	01177563          	bleu	a7,a4,1c008e96 <__udivdi3+0x2d2>
1c008e90:	ffe58313          	addi	t1,a1,-2
1c008e94:	9736                	add	a4,a4,a3
1c008e96:	07c2                	slli	a5,a5,0x10
1c008e98:	6e41                	lui	t3,0x10
1c008e9a:	0067e7b3          	or	a5,a5,t1
1c008e9e:	fffe0593          	addi	a1,t3,-1 # ffff <_l1_preload_size+0xf67f>
1c008ea2:	00b7f6b3          	and	a3,a5,a1
1c008ea6:	41170733          	sub	a4,a4,a7
1c008eaa:	8df1                	and	a1,a1,a2
1c008eac:	0107d893          	srli	a7,a5,0x10
1c008eb0:	02b68333          	mul	t1,a3,a1
1c008eb4:	8241                	srli	a2,a2,0x10
1c008eb6:	02b885b3          	mul	a1,a7,a1
1c008eba:	8eae                	mv	t4,a1
1c008ebc:	42c68eb3          	p.mac	t4,a3,a2
1c008ec0:	01035693          	srli	a3,t1,0x10
1c008ec4:	96f6                	add	a3,a3,t4
1c008ec6:	02c888b3          	mul	a7,a7,a2
1c008eca:	00b6f363          	bleu	a1,a3,1c008ed0 <__udivdi3+0x30c>
1c008ece:	98f2                	add	a7,a7,t3
1c008ed0:	0106d613          	srli	a2,a3,0x10
1c008ed4:	98b2                	add	a7,a7,a2
1c008ed6:	03176063          	bltu	a4,a7,1c008ef6 <__udivdi3+0x332>
1c008eda:	db1713e3          	bne	a4,a7,1c008c80 <__udivdi3+0xbc>
1c008ede:	6741                	lui	a4,0x10
1c008ee0:	177d                	addi	a4,a4,-1
1c008ee2:	8ef9                	and	a3,a3,a4
1c008ee4:	06c2                	slli	a3,a3,0x10
1c008ee6:	00e37333          	and	t1,t1,a4
1c008eea:	01051533          	sll	a0,a0,a6
1c008eee:	969a                	add	a3,a3,t1
1c008ef0:	4801                	li	a6,0
1c008ef2:	e2d57de3          	bleu	a3,a0,1c008d2c <__udivdi3+0x168>
1c008ef6:	17fd                	addi	a5,a5,-1
1c008ef8:	b361                	j	1c008c80 <__udivdi3+0xbc>
1c008efa:	4801                	li	a6,0
1c008efc:	4781                	li	a5,0
1c008efe:	b53d                	j	1c008d2c <__udivdi3+0x168>

1c008f00 <__umoddi3>:
1c008f00:	88b2                	mv	a7,a2
1c008f02:	8736                	mv	a4,a3
1c008f04:	87aa                	mv	a5,a0
1c008f06:	882e                	mv	a6,a1
1c008f08:	1a069963          	bnez	a3,1c0090ba <__umoddi3+0x1ba>
1c008f0c:	1c0016b7          	lui	a3,0x1c001
1c008f10:	cb068693          	addi	a3,a3,-848 # 1c000cb0 <__clz_tab>
1c008f14:	0ac5f463          	bleu	a2,a1,1c008fbc <__umoddi3+0xbc>
1c008f18:	6341                	lui	t1,0x10
1c008f1a:	08667a63          	bleu	t1,a2,1c008fae <__umoddi3+0xae>
1c008f1e:	0ff00313          	li	t1,255
1c008f22:	00c37363          	bleu	a2,t1,1c008f28 <__umoddi3+0x28>
1c008f26:	4721                	li	a4,8
1c008f28:	00e65333          	srl	t1,a2,a4
1c008f2c:	969a                	add	a3,a3,t1
1c008f2e:	0006c683          	lbu	a3,0(a3)
1c008f32:	02000313          	li	t1,32
1c008f36:	9736                	add	a4,a4,a3
1c008f38:	40e30333          	sub	t1,t1,a4
1c008f3c:	00030c63          	beqz	t1,1c008f54 <__umoddi3+0x54>
1c008f40:	006595b3          	sll	a1,a1,t1
1c008f44:	00e55733          	srl	a4,a0,a4
1c008f48:	006618b3          	sll	a7,a2,t1
1c008f4c:	00b76833          	or	a6,a4,a1
1c008f50:	006517b3          	sll	a5,a0,t1
1c008f54:	0108d613          	srli	a2,a7,0x10
1c008f58:	02c87733          	remu	a4,a6,a2
1c008f5c:	1008d533          	p.exthz	a0,a7
1c008f60:	0107d693          	srli	a3,a5,0x10
1c008f64:	02c85833          	divu	a6,a6,a2
1c008f68:	0742                	slli	a4,a4,0x10
1c008f6a:	8ed9                	or	a3,a3,a4
1c008f6c:	03050833          	mul	a6,a0,a6
1c008f70:	0106f863          	bleu	a6,a3,1c008f80 <__umoddi3+0x80>
1c008f74:	96c6                	add	a3,a3,a7
1c008f76:	0116e563          	bltu	a3,a7,1c008f80 <__umoddi3+0x80>
1c008f7a:	0106f363          	bleu	a6,a3,1c008f80 <__umoddi3+0x80>
1c008f7e:	96c6                	add	a3,a3,a7
1c008f80:	410686b3          	sub	a3,a3,a6
1c008f84:	02c6f733          	remu	a4,a3,a2
1c008f88:	02c6d6b3          	divu	a3,a3,a2
1c008f8c:	df0727b3          	p.insert	a5,a4,15,16
1c008f90:	02d506b3          	mul	a3,a0,a3
1c008f94:	00d7f863          	bleu	a3,a5,1c008fa4 <__umoddi3+0xa4>
1c008f98:	97c6                	add	a5,a5,a7
1c008f9a:	0117e563          	bltu	a5,a7,1c008fa4 <__umoddi3+0xa4>
1c008f9e:	00d7f363          	bleu	a3,a5,1c008fa4 <__umoddi3+0xa4>
1c008fa2:	97c6                	add	a5,a5,a7
1c008fa4:	8f95                	sub	a5,a5,a3
1c008fa6:	0067d533          	srl	a0,a5,t1
1c008faa:	4581                	li	a1,0
1c008fac:	8082                	ret
1c008fae:	01000337          	lui	t1,0x1000
1c008fb2:	4741                	li	a4,16
1c008fb4:	f6666ae3          	bltu	a2,t1,1c008f28 <__umoddi3+0x28>
1c008fb8:	4761                	li	a4,24
1c008fba:	b7bd                	j	1c008f28 <__umoddi3+0x28>
1c008fbc:	e601                	bnez	a2,1c008fc4 <__umoddi3+0xc4>
1c008fbe:	4605                	li	a2,1
1c008fc0:	031658b3          	divu	a7,a2,a7
1c008fc4:	6641                	lui	a2,0x10
1c008fc6:	06c8ff63          	bleu	a2,a7,1c009044 <__umoddi3+0x144>
1c008fca:	0ff00613          	li	a2,255
1c008fce:	01167363          	bleu	a7,a2,1c008fd4 <__umoddi3+0xd4>
1c008fd2:	4721                	li	a4,8
1c008fd4:	00e8d633          	srl	a2,a7,a4
1c008fd8:	96b2                	add	a3,a3,a2
1c008fda:	0006c603          	lbu	a2,0(a3)
1c008fde:	02000313          	li	t1,32
1c008fe2:	963a                	add	a2,a2,a4
1c008fe4:	40c30333          	sub	t1,t1,a2
1c008fe8:	06031563          	bnez	t1,1c009052 <__umoddi3+0x152>
1c008fec:	411585b3          	sub	a1,a1,a7
1c008ff0:	0108d713          	srli	a4,a7,0x10
1c008ff4:	1008d533          	p.exthz	a0,a7
1c008ff8:	0107d613          	srli	a2,a5,0x10
1c008ffc:	02e5f6b3          	remu	a3,a1,a4
1c009000:	02e5d5b3          	divu	a1,a1,a4
1c009004:	06c2                	slli	a3,a3,0x10
1c009006:	8ed1                	or	a3,a3,a2
1c009008:	02b505b3          	mul	a1,a0,a1
1c00900c:	00b6f863          	bleu	a1,a3,1c00901c <__umoddi3+0x11c>
1c009010:	96c6                	add	a3,a3,a7
1c009012:	0116e563          	bltu	a3,a7,1c00901c <__umoddi3+0x11c>
1c009016:	00b6f363          	bleu	a1,a3,1c00901c <__umoddi3+0x11c>
1c00901a:	96c6                	add	a3,a3,a7
1c00901c:	40b685b3          	sub	a1,a3,a1
1c009020:	02e5f6b3          	remu	a3,a1,a4
1c009024:	02e5d5b3          	divu	a1,a1,a4
1c009028:	df06a7b3          	p.insert	a5,a3,15,16
1c00902c:	02b505b3          	mul	a1,a0,a1
1c009030:	00b7f863          	bleu	a1,a5,1c009040 <__umoddi3+0x140>
1c009034:	97c6                	add	a5,a5,a7
1c009036:	0117e563          	bltu	a5,a7,1c009040 <__umoddi3+0x140>
1c00903a:	00b7f363          	bleu	a1,a5,1c009040 <__umoddi3+0x140>
1c00903e:	97c6                	add	a5,a5,a7
1c009040:	8f8d                	sub	a5,a5,a1
1c009042:	b795                	j	1c008fa6 <__umoddi3+0xa6>
1c009044:	01000637          	lui	a2,0x1000
1c009048:	4741                	li	a4,16
1c00904a:	f8c8e5e3          	bltu	a7,a2,1c008fd4 <__umoddi3+0xd4>
1c00904e:	4761                	li	a4,24
1c009050:	b751                	j	1c008fd4 <__umoddi3+0xd4>
1c009052:	006898b3          	sll	a7,a7,t1
1c009056:	00c5d733          	srl	a4,a1,a2
1c00905a:	006517b3          	sll	a5,a0,t1
1c00905e:	00c55633          	srl	a2,a0,a2
1c009062:	006595b3          	sll	a1,a1,t1
1c009066:	0108d513          	srli	a0,a7,0x10
1c00906a:	8dd1                	or	a1,a1,a2
1c00906c:	02a77633          	remu	a2,a4,a0
1c009070:	1008d833          	p.exthz	a6,a7
1c009074:	0105d693          	srli	a3,a1,0x10
1c009078:	02a75733          	divu	a4,a4,a0
1c00907c:	0642                	slli	a2,a2,0x10
1c00907e:	8ed1                	or	a3,a3,a2
1c009080:	02e80733          	mul	a4,a6,a4
1c009084:	00e6f863          	bleu	a4,a3,1c009094 <__umoddi3+0x194>
1c009088:	96c6                	add	a3,a3,a7
1c00908a:	0116e563          	bltu	a3,a7,1c009094 <__umoddi3+0x194>
1c00908e:	00e6f363          	bleu	a4,a3,1c009094 <__umoddi3+0x194>
1c009092:	96c6                	add	a3,a3,a7
1c009094:	8e99                	sub	a3,a3,a4
1c009096:	02a6f733          	remu	a4,a3,a0
1c00909a:	02a6d6b3          	divu	a3,a3,a0
1c00909e:	df0725b3          	p.insert	a1,a4,15,16
1c0090a2:	02d806b3          	mul	a3,a6,a3
1c0090a6:	00d5f863          	bleu	a3,a1,1c0090b6 <__umoddi3+0x1b6>
1c0090aa:	95c6                	add	a1,a1,a7
1c0090ac:	0115e563          	bltu	a1,a7,1c0090b6 <__umoddi3+0x1b6>
1c0090b0:	00d5f363          	bleu	a3,a1,1c0090b6 <__umoddi3+0x1b6>
1c0090b4:	95c6                	add	a1,a1,a7
1c0090b6:	8d95                	sub	a1,a1,a3
1c0090b8:	bf25                	j	1c008ff0 <__umoddi3+0xf0>
1c0090ba:	eed5e9e3          	bltu	a1,a3,1c008fac <__umoddi3+0xac>
1c0090be:	6741                	lui	a4,0x10
1c0090c0:	04e6f563          	bleu	a4,a3,1c00910a <__umoddi3+0x20a>
1c0090c4:	0ff00e93          	li	t4,255
1c0090c8:	00deb733          	sltu	a4,t4,a3
1c0090cc:	070e                	slli	a4,a4,0x3
1c0090ce:	1c001337          	lui	t1,0x1c001
1c0090d2:	00e6d8b3          	srl	a7,a3,a4
1c0090d6:	cb030313          	addi	t1,t1,-848 # 1c000cb0 <__clz_tab>
1c0090da:	989a                	add	a7,a7,t1
1c0090dc:	0008ce83          	lbu	t4,0(a7)
1c0090e0:	02000e13          	li	t3,32
1c0090e4:	9eba                	add	t4,t4,a4
1c0090e6:	41de0e33          	sub	t3,t3,t4
1c0090ea:	020e1763          	bnez	t3,1c009118 <__umoddi3+0x218>
1c0090ee:	00b6e463          	bltu	a3,a1,1c0090f6 <__umoddi3+0x1f6>
1c0090f2:	00c56963          	bltu	a0,a2,1c009104 <__umoddi3+0x204>
1c0090f6:	40c507b3          	sub	a5,a0,a2
1c0090fa:	8d95                	sub	a1,a1,a3
1c0090fc:	00f53533          	sltu	a0,a0,a5
1c009100:	40a58833          	sub	a6,a1,a0
1c009104:	853e                	mv	a0,a5
1c009106:	85c2                	mv	a1,a6
1c009108:	b555                	j	1c008fac <__umoddi3+0xac>
1c00910a:	010008b7          	lui	a7,0x1000
1c00910e:	4741                	li	a4,16
1c009110:	fb16efe3          	bltu	a3,a7,1c0090ce <__umoddi3+0x1ce>
1c009114:	4761                	li	a4,24
1c009116:	bf65                	j	1c0090ce <__umoddi3+0x1ce>
1c009118:	01d65733          	srl	a4,a2,t4
1c00911c:	01c696b3          	sll	a3,a3,t3
1c009120:	8ed9                	or	a3,a3,a4
1c009122:	01d5d7b3          	srl	a5,a1,t4
1c009126:	0106d813          	srli	a6,a3,0x10
1c00912a:	0307f333          	remu	t1,a5,a6
1c00912e:	1006d733          	p.exthz	a4,a3
1c009132:	01d558b3          	srl	a7,a0,t4
1c009136:	01c595b3          	sll	a1,a1,t3
1c00913a:	00b8e5b3          	or	a1,a7,a1
1c00913e:	0105d893          	srli	a7,a1,0x10
1c009142:	01c61633          	sll	a2,a2,t3
1c009146:	01c51533          	sll	a0,a0,t3
1c00914a:	0307d7b3          	divu	a5,a5,a6
1c00914e:	0342                	slli	t1,t1,0x10
1c009150:	011368b3          	or	a7,t1,a7
1c009154:	02f70f33          	mul	t5,a4,a5
1c009158:	833e                	mv	t1,a5
1c00915a:	01e8fc63          	bleu	t5,a7,1c009172 <__umoddi3+0x272>
1c00915e:	98b6                	add	a7,a7,a3
1c009160:	fff78313          	addi	t1,a5,-1 # ffffff <__l1_heap_size+0xf00983>
1c009164:	00d8e763          	bltu	a7,a3,1c009172 <__umoddi3+0x272>
1c009168:	01e8f563          	bleu	t5,a7,1c009172 <__umoddi3+0x272>
1c00916c:	ffe78313          	addi	t1,a5,-2
1c009170:	98b6                	add	a7,a7,a3
1c009172:	41e888b3          	sub	a7,a7,t5
1c009176:	0308f7b3          	remu	a5,a7,a6
1c00917a:	0308d8b3          	divu	a7,a7,a6
1c00917e:	df07a5b3          	p.insert	a1,a5,15,16
1c009182:	03170733          	mul	a4,a4,a7
1c009186:	87c6                	mv	a5,a7
1c009188:	00e5fc63          	bleu	a4,a1,1c0091a0 <__umoddi3+0x2a0>
1c00918c:	95b6                	add	a1,a1,a3
1c00918e:	fff88793          	addi	a5,a7,-1 # ffffff <__l1_heap_size+0xf00983>
1c009192:	00d5e763          	bltu	a1,a3,1c0091a0 <__umoddi3+0x2a0>
1c009196:	00e5f563          	bleu	a4,a1,1c0091a0 <__umoddi3+0x2a0>
1c00919a:	ffe88793          	addi	a5,a7,-2
1c00919e:	95b6                	add	a1,a1,a3
1c0091a0:	0342                	slli	t1,t1,0x10
1c0091a2:	6f41                	lui	t5,0x10
1c0091a4:	00f36333          	or	t1,t1,a5
1c0091a8:	ffff0793          	addi	a5,t5,-1 # ffff <_l1_preload_size+0xf67f>
1c0091ac:	00f37833          	and	a6,t1,a5
1c0091b0:	01035313          	srli	t1,t1,0x10
1c0091b4:	8ff1                	and	a5,a5,a2
1c0091b6:	02f808b3          	mul	a7,a6,a5
1c0091ba:	8d99                	sub	a1,a1,a4
1c0091bc:	01065713          	srli	a4,a2,0x10
1c0091c0:	02f307b3          	mul	a5,t1,a5
1c0091c4:	8fbe                	mv	t6,a5
1c0091c6:	42e80fb3          	p.mac	t6,a6,a4
1c0091ca:	0108d813          	srli	a6,a7,0x10
1c0091ce:	987e                	add	a6,a6,t6
1c0091d0:	02e30333          	mul	t1,t1,a4
1c0091d4:	00f87363          	bleu	a5,a6,1c0091da <__umoddi3+0x2da>
1c0091d8:	937a                	add	t1,t1,t5
1c0091da:	01085713          	srli	a4,a6,0x10
1c0091de:	933a                	add	t1,t1,a4
1c0091e0:	6741                	lui	a4,0x10
1c0091e2:	177d                	addi	a4,a4,-1
1c0091e4:	00e87833          	and	a6,a6,a4
1c0091e8:	0842                	slli	a6,a6,0x10
1c0091ea:	00e8f733          	and	a4,a7,a4
1c0091ee:	9742                	add	a4,a4,a6
1c0091f0:	0065e663          	bltu	a1,t1,1c0091fc <__umoddi3+0x2fc>
1c0091f4:	00659d63          	bne	a1,t1,1c00920e <__umoddi3+0x30e>
1c0091f8:	00e57b63          	bleu	a4,a0,1c00920e <__umoddi3+0x30e>
1c0091fc:	40c70633          	sub	a2,a4,a2
1c009200:	00c73733          	sltu	a4,a4,a2
1c009204:	40d30333          	sub	t1,t1,a3
1c009208:	40e30333          	sub	t1,t1,a4
1c00920c:	8732                	mv	a4,a2
1c00920e:	40e50733          	sub	a4,a0,a4
1c009212:	00e53533          	sltu	a0,a0,a4
1c009216:	406585b3          	sub	a1,a1,t1
1c00921a:	8d89                	sub	a1,a1,a0
1c00921c:	01d597b3          	sll	a5,a1,t4
1c009220:	01c75733          	srl	a4,a4,t3
1c009224:	00e7e533          	or	a0,a5,a4
1c009228:	01c5d5b3          	srl	a1,a1,t3
1c00922c:	b341                	j	1c008fac <__umoddi3+0xac>

1c00922e <__adddf3>:
1c00922e:	00202ef3          	csrr	t4,0x2
1c009232:	e60597b3          	p.extractu	a5,a1,19,0
1c009236:	00379893          	slli	a7,a5,0x3
1c00923a:	01d55793          	srli	a5,a0,0x1d
1c00923e:	d5459f33          	p.extractu	t5,a1,10,20
1c009242:	d5469333          	p.extractu	t1,a3,10,20
1c009246:	0117e7b3          	or	a5,a5,a7
1c00924a:	e60698b3          	p.extractu	a7,a3,19,0
1c00924e:	01d65713          	srli	a4,a2,0x1d
1c009252:	088e                	slli	a7,a7,0x3
1c009254:	00361813          	slli	a6,a2,0x3
1c009258:	81fd                	srli	a1,a1,0x1f
1c00925a:	406f0633          	sub	a2,t5,t1
1c00925e:	82fd                	srli	a3,a3,0x1f
1c009260:	050e                	slli	a0,a0,0x3
1c009262:	01176733          	or	a4,a4,a7
1c009266:	8e32                	mv	t3,a2
1c009268:	38d59b63          	bne	a1,a3,1c0095fe <__adddf3+0x3d0>
1c00926c:	12c05263          	blez	a2,1c009390 <__adddf3+0x162>
1c009270:	08031663          	bnez	t1,1c0092fc <__adddf3+0xce>
1c009274:	010766b3          	or	a3,a4,a6
1c009278:	e295                	bnez	a3,1c00929c <__adddf3+0x6e>
1c00927a:	7ff00713          	li	a4,2047
1c00927e:	38e61b63          	bne	a2,a4,1c009614 <__adddf3+0x3e6>
1c009282:	00a7e733          	or	a4,a5,a0
1c009286:	76070763          	beqz	a4,1c0099f4 <__adddf3+0x7c6>
1c00928a:	0167d713          	srli	a4,a5,0x16
1c00928e:	7ff00313          	li	t1,2047
1c009292:	4e01                	li	t3,0
1c009294:	40070b63          	beqz	a4,1c0096aa <__adddf3+0x47c>
1c009298:	4881                	li	a7,0
1c00929a:	ac39                	j	1c0094b8 <__adddf3+0x28a>
1c00929c:	167d                	addi	a2,a2,-1
1c00929e:	e221                	bnez	a2,1c0092de <__adddf3+0xb0>
1c0092a0:	982a                	add	a6,a6,a0
1c0092a2:	97ba                	add	a5,a5,a4
1c0092a4:	00a83533          	sltu	a0,a6,a0
1c0092a8:	00a78733          	add	a4,a5,a0
1c0092ac:	4305                	li	t1,1
1c0092ae:	00871793          	slli	a5,a4,0x8
1c0092b2:	0e07d963          	bgez	a5,1c0093a4 <__adddf3+0x176>
1c0092b6:	0305                	addi	t1,t1,1
1c0092b8:	7ff00793          	li	a5,2047
1c0092bc:	30f30f63          	beq	t1,a5,1c0095da <__adddf3+0x3ac>
1c0092c0:	c17737b3          	p.bclr	a5,a4,0,23
1c0092c4:	00185693          	srli	a3,a6,0x1
1c0092c8:	fc183833          	p.bclr	a6,a6,30,1
1c0092cc:	01f79513          	slli	a0,a5,0x1f
1c0092d0:	0106e833          	or	a6,a3,a6
1c0092d4:	01056533          	or	a0,a0,a6
1c0092d8:	8385                	srli	a5,a5,0x1
1c0092da:	4e01                	li	t3,0
1c0092dc:	bf75                	j	1c009298 <__adddf3+0x6a>
1c0092de:	7ff00693          	li	a3,2047
1c0092e2:	02de1c63          	bne	t3,a3,1c00931a <__adddf3+0xec>
1c0092e6:	00a7e733          	or	a4,a5,a0
1c0092ea:	70070563          	beqz	a4,1c0099f4 <__adddf3+0x7c6>
1c0092ee:	00979313          	slli	t1,a5,0x9
1c0092f2:	6e035d63          	bgez	t1,1c0099ec <__adddf3+0x7be>
1c0092f6:	7ff00313          	li	t1,2047
1c0092fa:	b7c5                	j	1c0092da <__adddf3+0xac>
1c0092fc:	7ff00693          	li	a3,2047
1c009300:	00df1b63          	bne	t5,a3,1c009316 <__adddf3+0xe8>
1c009304:	00a7e733          	or	a4,a5,a0
1c009308:	6e070663          	beqz	a4,1c0099f4 <__adddf3+0x7c6>
1c00930c:	00979613          	slli	a2,a5,0x9
1c009310:	7ff00313          	li	t1,2047
1c009314:	a0e1                	j	1c0093dc <__adddf3+0x1ae>
1c009316:	c1774733          	p.bset	a4,a4,0,23
1c00931a:	03800693          	li	a3,56
1c00931e:	06c6c463          	blt	a3,a2,1c009386 <__adddf3+0x158>
1c009322:	46fd                	li	a3,31
1c009324:	02c6cc63          	blt	a3,a2,1c00935c <__adddf3+0x12e>
1c009328:	02000893          	li	a7,32
1c00932c:	40c888b3          	sub	a7,a7,a2
1c009330:	00c85333          	srl	t1,a6,a2
1c009334:	011716b3          	sll	a3,a4,a7
1c009338:	01181833          	sll	a6,a6,a7
1c00933c:	0066e6b3          	or	a3,a3,t1
1c009340:	01003833          	snez	a6,a6
1c009344:	0106e833          	or	a6,a3,a6
1c009348:	00c75633          	srl	a2,a4,a2
1c00934c:	982a                	add	a6,a6,a0
1c00934e:	963e                	add	a2,a2,a5
1c009350:	00a837b3          	sltu	a5,a6,a0
1c009354:	00f60733          	add	a4,a2,a5
1c009358:	837a                	mv	t1,t5
1c00935a:	bf91                	j	1c0092ae <__adddf3+0x80>
1c00935c:	02000313          	li	t1,32
1c009360:	00c756b3          	srl	a3,a4,a2
1c009364:	4881                	li	a7,0
1c009366:	00660863          	beq	a2,t1,1c009376 <__adddf3+0x148>
1c00936a:	04000893          	li	a7,64
1c00936e:	40c88633          	sub	a2,a7,a2
1c009372:	00c718b3          	sll	a7,a4,a2
1c009376:	0108e833          	or	a6,a7,a6
1c00937a:	01003833          	snez	a6,a6
1c00937e:	0106e833          	or	a6,a3,a6
1c009382:	4601                	li	a2,0
1c009384:	b7e1                	j	1c00934c <__adddf3+0x11e>
1c009386:	01076833          	or	a6,a4,a6
1c00938a:	01003833          	snez	a6,a6
1c00938e:	bfd5                	j	1c009382 <__adddf3+0x154>
1c009390:	c66d                	beqz	a2,1c00947a <__adddf3+0x24c>
1c009392:	040f1963          	bnez	t5,1c0093e4 <__adddf3+0x1b6>
1c009396:	00a7e6b3          	or	a3,a5,a0
1c00939a:	ea89                	bnez	a3,1c0093ac <__adddf3+0x17e>
1c00939c:	7ff00793          	li	a5,2047
1c0093a0:	04f30663          	beq	t1,a5,1c0093ec <__adddf3+0x1be>
1c0093a4:	04031c63          	bnez	t1,1c0093fc <__adddf3+0x1ce>
1c0093a8:	8542                	mv	a0,a6
1c0093aa:	a999                	j	1c009800 <__adddf3+0x5d2>
1c0093ac:	01f63a63          	p.bneimm	a2,-1,1c0093c0 <__adddf3+0x192>
1c0093b0:	010506b3          	add	a3,a0,a6
1c0093b4:	97ba                	add	a5,a5,a4
1c0093b6:	0106b733          	sltu	a4,a3,a6
1c0093ba:	973e                	add	a4,a4,a5
1c0093bc:	8836                	mv	a6,a3
1c0093be:	bdc5                	j	1c0092ae <__adddf3+0x80>
1c0093c0:	7ff00693          	li	a3,2047
1c0093c4:	fff64613          	not	a2,a2
1c0093c8:	04d31163          	bne	t1,a3,1c00940a <__adddf3+0x1dc>
1c0093cc:	01076533          	or	a0,a4,a6
1c0093d0:	62050863          	beqz	a0,1c009a00 <__adddf3+0x7d2>
1c0093d4:	00971613          	slli	a2,a4,0x9
1c0093d8:	87ba                	mv	a5,a4
1c0093da:	8542                	mv	a0,a6
1c0093dc:	4e01                	li	t3,0
1c0093de:	2c065663          	bgez	a2,1c0096aa <__adddf3+0x47c>
1c0093e2:	bd5d                	j	1c009298 <__adddf3+0x6a>
1c0093e4:	7ff00693          	li	a3,2047
1c0093e8:	00d31d63          	bne	t1,a3,1c009402 <__adddf3+0x1d4>
1c0093ec:	01076533          	or	a0,a4,a6
1c0093f0:	60050863          	beqz	a0,1c009a00 <__adddf3+0x7d2>
1c0093f4:	00971793          	slli	a5,a4,0x9
1c0093f8:	6007d663          	bgez	a5,1c009a04 <__adddf3+0x7d6>
1c0093fc:	87ba                	mv	a5,a4
1c0093fe:	8542                	mv	a0,a6
1c009400:	bde9                	j	1c0092da <__adddf3+0xac>
1c009402:	40c00633          	neg	a2,a2
1c009406:	c177c7b3          	p.bset	a5,a5,0,23
1c00940a:	03800693          	li	a3,56
1c00940e:	06c6c263          	blt	a3,a2,1c009472 <__adddf3+0x244>
1c009412:	46fd                	li	a3,31
1c009414:	02c6cb63          	blt	a3,a2,1c00944a <__adddf3+0x21c>
1c009418:	02000893          	li	a7,32
1c00941c:	40c888b3          	sub	a7,a7,a2
1c009420:	00c55e33          	srl	t3,a0,a2
1c009424:	011796b3          	sll	a3,a5,a7
1c009428:	01151533          	sll	a0,a0,a7
1c00942c:	01c6e6b3          	or	a3,a3,t3
1c009430:	00a03533          	snez	a0,a0
1c009434:	8d55                	or	a0,a0,a3
1c009436:	00c7d633          	srl	a2,a5,a2
1c00943a:	9542                	add	a0,a0,a6
1c00943c:	00e607b3          	add	a5,a2,a4
1c009440:	01053733          	sltu	a4,a0,a6
1c009444:	973e                	add	a4,a4,a5
1c009446:	882a                	mv	a6,a0
1c009448:	b59d                	j	1c0092ae <__adddf3+0x80>
1c00944a:	02000e13          	li	t3,32
1c00944e:	00c7d6b3          	srl	a3,a5,a2
1c009452:	4881                	li	a7,0
1c009454:	01c60863          	beq	a2,t3,1c009464 <__adddf3+0x236>
1c009458:	04000893          	li	a7,64
1c00945c:	40c88633          	sub	a2,a7,a2
1c009460:	00c798b3          	sll	a7,a5,a2
1c009464:	00a8e533          	or	a0,a7,a0
1c009468:	00a03533          	snez	a0,a0
1c00946c:	8d55                	or	a0,a0,a3
1c00946e:	4601                	li	a2,0
1c009470:	b7e9                	j	1c00943a <__adddf3+0x20c>
1c009472:	8d5d                	or	a0,a0,a5
1c009474:	00a03533          	snez	a0,a0
1c009478:	bfdd                	j	1c00946e <__adddf3+0x240>
1c00947a:	001f0693          	addi	a3,t5,1
1c00947e:	e8b6b8b3          	p.bclr	a7,a3,20,11
1c009482:	4605                	li	a2,1
1c009484:	0f164663          	blt	a2,a7,1c009570 <__adddf3+0x342>
1c009488:	00a7e6b3          	or	a3,a5,a0
1c00948c:	040f1163          	bnez	t5,1c0094ce <__adddf3+0x2a0>
1c009490:	de81                	beqz	a3,1c0093a8 <__adddf3+0x17a>
1c009492:	010766b3          	or	a3,a4,a6
1c009496:	52068663          	beqz	a3,1c0099c2 <__adddf3+0x794>
1c00949a:	982a                	add	a6,a6,a0
1c00949c:	97ba                	add	a5,a5,a4
1c00949e:	00a83533          	sltu	a0,a6,a0
1c0094a2:	00a78733          	add	a4,a5,a0
1c0094a6:	00871313          	slli	t1,a4,0x8
1c0094aa:	ee035fe3          	bgez	t1,1c0093a8 <__adddf3+0x17a>
1c0094ae:	c17737b3          	p.bclr	a5,a4,0,23
1c0094b2:	8542                	mv	a0,a6
1c0094b4:	4881                	li	a7,0
1c0094b6:	4305                	li	t1,1
1c0094b8:	f8353733          	p.bclr	a4,a0,28,3
1c0094bc:	e775                	bnez	a4,1c0095a8 <__adddf3+0x37a>
1c0094be:	040e0b63          	beqz	t3,1c009514 <__adddf3+0x2e6>
1c0094c2:	fc18b733          	p.bclr	a4,a7,30,1
1c0094c6:	c739                	beqz	a4,1c009514 <__adddf3+0x2e6>
1c0094c8:	c018c8b3          	p.bset	a7,a7,0,1
1c0094cc:	a0a1                	j	1c009514 <__adddf3+0x2e6>
1c0094ce:	7ff00613          	li	a2,2047
1c0094d2:	08cf1a63          	bne	t5,a2,1c009566 <__adddf3+0x338>
1c0094d6:	58068963          	beqz	a3,1c009a68 <__adddf3+0x83a>
1c0094da:	004008b7          	lui	a7,0x400
1c0094de:	0117f8b3          	and	a7,a5,a7
1c0094e2:	0018b893          	seqz	a7,a7
1c0094e6:	0892                	slli	a7,a7,0x4
1c0094e8:	01e31c63          	bne	t1,t5,1c009500 <__adddf3+0x2d2>
1c0094ec:	01076633          	or	a2,a4,a6
1c0094f0:	c611                	beqz	a2,1c0094fc <__adddf3+0x2ce>
1c0094f2:	00971613          	slli	a2,a4,0x9
1c0094f6:	00064363          	bltz	a2,1c0094fc <__adddf3+0x2ce>
1c0094fa:	48c1                	li	a7,16
1c0094fc:	4e068563          	beqz	a3,1c0099e6 <__adddf3+0x7b8>
1c009500:	01076733          	or	a4,a4,a6
1c009504:	3c070263          	beqz	a4,1c0098c8 <__adddf3+0x69a>
1c009508:	004007b7          	lui	a5,0x400
1c00950c:	4501                	li	a0,0
1c00950e:	7ff00313          	li	t1,2047
1c009512:	4581                	li	a1,0
1c009514:	00879713          	slli	a4,a5,0x8
1c009518:	00075963          	bgez	a4,1c00952a <__adddf3+0x2fc>
1c00951c:	0305                	addi	t1,t1,1
1c00951e:	7ff00713          	li	a4,2047
1c009522:	50e30e63          	beq	t1,a4,1c009a3e <__adddf3+0x810>
1c009526:	c177b7b3          	p.bclr	a5,a5,0,23
1c00952a:	01d79713          	slli	a4,a5,0x1d
1c00952e:	810d                	srli	a0,a0,0x3
1c009530:	7ff00693          	li	a3,2047
1c009534:	8f49                	or	a4,a4,a0
1c009536:	838d                	srli	a5,a5,0x3
1c009538:	00d31963          	bne	t1,a3,1c00954a <__adddf3+0x31c>
1c00953c:	8f5d                	or	a4,a4,a5
1c00953e:	4781                	li	a5,0
1c009540:	c709                	beqz	a4,1c00954a <__adddf3+0x31c>
1c009542:	000807b7          	lui	a5,0x80
1c009546:	4701                	li	a4,0
1c009548:	4581                	li	a1,0
1c00954a:	4681                	li	a3,0
1c00954c:	e607a6b3          	p.insert	a3,a5,19,0
1c009550:	d54326b3          	p.insert	a3,t1,10,20
1c009554:	c1f5a6b3          	p.insert	a3,a1,0,31
1c009558:	853a                	mv	a0,a4
1c00955a:	85b6                	mv	a1,a3
1c00955c:	50088d63          	beqz	a7,1c009a76 <__adddf3+0x848>
1c009560:	0018a073          	csrs	0x1,a7
1c009564:	8082                	ret
1c009566:	4881                	li	a7,0
1c009568:	f8c31ae3          	bne	t1,a2,1c0094fc <__adddf3+0x2ce>
1c00956c:	4881                	li	a7,0
1c00956e:	bfbd                	j	1c0094ec <__adddf3+0x2be>
1c009570:	7ff00613          	li	a2,2047
1c009574:	02c68063          	beq	a3,a2,1c009594 <__adddf3+0x366>
1c009578:	982a                	add	a6,a6,a0
1c00957a:	00a83533          	sltu	a0,a6,a0
1c00957e:	97ba                	add	a5,a5,a4
1c009580:	97aa                	add	a5,a5,a0
1c009582:	01f79513          	slli	a0,a5,0x1f
1c009586:	00185813          	srli	a6,a6,0x1
1c00958a:	01056533          	or	a0,a0,a6
1c00958e:	8385                	srli	a5,a5,0x1
1c009590:	8336                	mv	t1,a3
1c009592:	b319                	j	1c009298 <__adddf3+0x6a>
1c009594:	040e8063          	beqz	t4,1c0095d4 <__adddf3+0x3a6>
1c009598:	023ebb63          	p.bneimm	t4,3,1c0095ce <__adddf3+0x3a0>
1c00959c:	cd85                	beqz	a1,1c0095d4 <__adddf3+0x3a6>
1c00959e:	57fd                	li	a5,-1
1c0095a0:	557d                	li	a0,-1
1c0095a2:	7fe00313          	li	t1,2046
1c0095a6:	4895                	li	a7,5
1c0095a8:	c008c8b3          	p.bset	a7,a7,0,0
1c0095ac:	482ea663          	p.beqimm	t4,2,1c009a38 <__adddf3+0x80a>
1c0095b0:	463eaf63          	p.beqimm	t4,3,1c009a2e <__adddf3+0x800>
1c0095b4:	f00e95e3          	bnez	t4,1c0094be <__adddf3+0x290>
1c0095b8:	f6453733          	p.bclr	a4,a0,27,4
1c0095bc:	f04721e3          	p.beqimm	a4,4,1c0094be <__adddf3+0x290>
1c0095c0:	00450713          	addi	a4,a0,4
1c0095c4:	00a73533          	sltu	a0,a4,a0
1c0095c8:	97aa                	add	a5,a5,a0
1c0095ca:	853a                	mv	a0,a4
1c0095cc:	bdcd                	j	1c0094be <__adddf3+0x290>
1c0095ce:	fc2eb8e3          	p.bneimm	t4,2,1c00959e <__adddf3+0x370>
1c0095d2:	d5f1                	beqz	a1,1c00959e <__adddf3+0x370>
1c0095d4:	7ff00313          	li	t1,2047
1c0095d8:	a839                	j	1c0095f6 <__adddf3+0x3c8>
1c0095da:	000e8e63          	beqz	t4,1c0095f6 <__adddf3+0x3c8>
1c0095de:	003eb963          	p.bneimm	t4,3,1c0095f0 <__adddf3+0x3c2>
1c0095e2:	c991                	beqz	a1,1c0095f6 <__adddf3+0x3c8>
1c0095e4:	57fd                	li	a5,-1
1c0095e6:	557d                	li	a0,-1
1c0095e8:	7fe00313          	li	t1,2046
1c0095ec:	4e01                	li	t3,0
1c0095ee:	bf65                	j	1c0095a6 <__adddf3+0x378>
1c0095f0:	fe2ebae3          	p.bneimm	t4,2,1c0095e4 <__adddf3+0x3b6>
1c0095f4:	d9e5                	beqz	a1,1c0095e4 <__adddf3+0x3b6>
1c0095f6:	4781                	li	a5,0
1c0095f8:	4501                	li	a0,0
1c0095fa:	4895                	li	a7,5
1c0095fc:	bf21                	j	1c009514 <__adddf3+0x2e6>
1c0095fe:	0ec05563          	blez	a2,1c0096e8 <__adddf3+0x4ba>
1c009602:	08031563          	bnez	t1,1c00968c <__adddf3+0x45e>
1c009606:	010766b3          	or	a3,a4,a6
1c00960a:	e699                	bnez	a3,1c009618 <__adddf3+0x3ea>
1c00960c:	7ff00713          	li	a4,2047
1c009610:	cce60be3          	beq	a2,a4,1c0092e6 <__adddf3+0xb8>
1c009614:	8332                	mv	t1,a2
1c009616:	b1d1                	j	1c0092da <__adddf3+0xac>
1c009618:	167d                	addi	a2,a2,-1
1c00961a:	e215                	bnez	a2,1c00963e <__adddf3+0x410>
1c00961c:	41050833          	sub	a6,a0,a6
1c009620:	8f99                	sub	a5,a5,a4
1c009622:	01053533          	sltu	a0,a0,a6
1c009626:	40a78733          	sub	a4,a5,a0
1c00962a:	4305                	li	t1,1
1c00962c:	00871613          	slli	a2,a4,0x8
1c009630:	d6065ae3          	bgez	a2,1c0093a4 <__adddf3+0x176>
1c009634:	d1773633          	p.bclr	a2,a4,8,23
1c009638:	88c2                	mv	a7,a6
1c00963a:	8f1a                	mv	t5,t1
1c00963c:	a4e9                	j	1c009906 <__adddf3+0x6d8>
1c00963e:	7ff00693          	li	a3,2047
1c009642:	ccde01e3          	beq	t3,a3,1c009304 <__adddf3+0xd6>
1c009646:	03800693          	li	a3,56
1c00964a:	08c6ca63          	blt	a3,a2,1c0096de <__adddf3+0x4b0>
1c00964e:	46fd                	li	a3,31
1c009650:	06c6c263          	blt	a3,a2,1c0096b4 <__adddf3+0x486>
1c009654:	02000893          	li	a7,32
1c009658:	40c888b3          	sub	a7,a7,a2
1c00965c:	00c85333          	srl	t1,a6,a2
1c009660:	011716b3          	sll	a3,a4,a7
1c009664:	01181833          	sll	a6,a6,a7
1c009668:	0066e6b3          	or	a3,a3,t1
1c00966c:	01003833          	snez	a6,a6
1c009670:	0106e833          	or	a6,a3,a6
1c009674:	00c75633          	srl	a2,a4,a2
1c009678:	41050833          	sub	a6,a0,a6
1c00967c:	40c78633          	sub	a2,a5,a2
1c009680:	010537b3          	sltu	a5,a0,a6
1c009684:	40f60733          	sub	a4,a2,a5
1c009688:	837a                	mv	t1,t5
1c00968a:	b74d                	j	1c00962c <__adddf3+0x3fe>
1c00968c:	7ff00693          	li	a3,2047
1c009690:	00df1f63          	bne	t5,a3,1c0096ae <__adddf3+0x480>
1c009694:	00a7e733          	or	a4,a5,a0
1c009698:	34070e63          	beqz	a4,1c0099f4 <__adddf3+0x7c6>
1c00969c:	00979713          	slli	a4,a5,0x9
1c0096a0:	7ff00313          	li	t1,2047
1c0096a4:	4e01                	li	t3,0
1c0096a6:	be0749e3          	bltz	a4,1c009298 <__adddf3+0x6a>
1c0096aa:	48c1                	li	a7,16
1c0096ac:	b531                	j	1c0094b8 <__adddf3+0x28a>
1c0096ae:	c1774733          	p.bset	a4,a4,0,23
1c0096b2:	bf51                	j	1c009646 <__adddf3+0x418>
1c0096b4:	02000313          	li	t1,32
1c0096b8:	00c756b3          	srl	a3,a4,a2
1c0096bc:	4881                	li	a7,0
1c0096be:	00660863          	beq	a2,t1,1c0096ce <__adddf3+0x4a0>
1c0096c2:	04000893          	li	a7,64
1c0096c6:	40c88633          	sub	a2,a7,a2
1c0096ca:	00c718b3          	sll	a7,a4,a2
1c0096ce:	0108e833          	or	a6,a7,a6
1c0096d2:	01003833          	snez	a6,a6
1c0096d6:	0106e833          	or	a6,a3,a6
1c0096da:	4601                	li	a2,0
1c0096dc:	bf71                	j	1c009678 <__adddf3+0x44a>
1c0096de:	01076833          	or	a6,a4,a6
1c0096e2:	01003833          	snez	a6,a6
1c0096e6:	bfd5                	j	1c0096da <__adddf3+0x4ac>
1c0096e8:	c66d                	beqz	a2,1c0097d2 <__adddf3+0x5a4>
1c0096ea:	040f1663          	bnez	t5,1c009736 <__adddf3+0x508>
1c0096ee:	00a7e5b3          	or	a1,a5,a0
1c0096f2:	e589                	bnez	a1,1c0096fc <__adddf3+0x4ce>
1c0096f4:	7ff00793          	li	a5,2047
1c0096f8:	85b6                	mv	a1,a3
1c0096fa:	b15d                	j	1c0093a0 <__adddf3+0x172>
1c0096fc:	01f63d63          	p.bneimm	a2,-1,1c009716 <__adddf3+0x4e8>
1c009700:	40a805b3          	sub	a1,a6,a0
1c009704:	40f707b3          	sub	a5,a4,a5
1c009708:	00b83733          	sltu	a4,a6,a1
1c00970c:	40e78733          	sub	a4,a5,a4
1c009710:	882e                	mv	a6,a1
1c009712:	85b6                	mv	a1,a3
1c009714:	bf21                	j	1c00962c <__adddf3+0x3fe>
1c009716:	7ff00593          	li	a1,2047
1c00971a:	fff64613          	not	a2,a2
1c00971e:	04b31063          	bne	t1,a1,1c00975e <__adddf3+0x530>
1c009722:	01076533          	or	a0,a4,a6
1c009726:	2e050263          	beqz	a0,1c009a0a <__adddf3+0x7dc>
1c00972a:	00971613          	slli	a2,a4,0x9
1c00972e:	87ba                	mv	a5,a4
1c009730:	8542                	mv	a0,a6
1c009732:	85b6                	mv	a1,a3
1c009734:	b165                	j	1c0093dc <__adddf3+0x1ae>
1c009736:	7ff00593          	li	a1,2047
1c00973a:	00b31e63          	bne	t1,a1,1c009756 <__adddf3+0x528>
1c00973e:	01076533          	or	a0,a4,a6
1c009742:	2c050463          	beqz	a0,1c009a0a <__adddf3+0x7dc>
1c009746:	00971793          	slli	a5,a4,0x9
1c00974a:	2c07d363          	bgez	a5,1c009a10 <__adddf3+0x7e2>
1c00974e:	87ba                	mv	a5,a4
1c009750:	8542                	mv	a0,a6
1c009752:	85b6                	mv	a1,a3
1c009754:	b659                	j	1c0092da <__adddf3+0xac>
1c009756:	40c00633          	neg	a2,a2
1c00975a:	c177c7b3          	p.bset	a5,a5,0,23
1c00975e:	03800593          	li	a1,56
1c009762:	06c5c463          	blt	a1,a2,1c0097ca <__adddf3+0x59c>
1c009766:	45fd                	li	a1,31
1c009768:	02c5cd63          	blt	a1,a2,1c0097a2 <__adddf3+0x574>
1c00976c:	02000893          	li	a7,32
1c009770:	40c888b3          	sub	a7,a7,a2
1c009774:	00c55e33          	srl	t3,a0,a2
1c009778:	011795b3          	sll	a1,a5,a7
1c00977c:	01151533          	sll	a0,a0,a7
1c009780:	01c5e5b3          	or	a1,a1,t3
1c009784:	00a03533          	snez	a0,a0
1c009788:	8d4d                	or	a0,a0,a1
1c00978a:	00c7d633          	srl	a2,a5,a2
1c00978e:	40a80533          	sub	a0,a6,a0
1c009792:	40c707b3          	sub	a5,a4,a2
1c009796:	00a83733          	sltu	a4,a6,a0
1c00979a:	40e78733          	sub	a4,a5,a4
1c00979e:	882a                	mv	a6,a0
1c0097a0:	bf8d                	j	1c009712 <__adddf3+0x4e4>
1c0097a2:	02000e13          	li	t3,32
1c0097a6:	00c7d5b3          	srl	a1,a5,a2
1c0097aa:	4881                	li	a7,0
1c0097ac:	01c60863          	beq	a2,t3,1c0097bc <__adddf3+0x58e>
1c0097b0:	04000893          	li	a7,64
1c0097b4:	40c88633          	sub	a2,a7,a2
1c0097b8:	00c798b3          	sll	a7,a5,a2
1c0097bc:	00a8e533          	or	a0,a7,a0
1c0097c0:	00a03533          	snez	a0,a0
1c0097c4:	8d4d                	or	a0,a0,a1
1c0097c6:	4601                	li	a2,0
1c0097c8:	b7d9                	j	1c00978e <__adddf3+0x560>
1c0097ca:	8d5d                	or	a0,a0,a5
1c0097cc:	00a03533          	snez	a0,a0
1c0097d0:	bfdd                	j	1c0097c6 <__adddf3+0x598>
1c0097d2:	001f0613          	addi	a2,t5,1
1c0097d6:	e8b63633          	p.bclr	a2,a2,20,11
1c0097da:	4885                	li	a7,1
1c0097dc:	10c8c063          	blt	a7,a2,1c0098dc <__adddf3+0x6ae>
1c0097e0:	00a7efb3          	or	t6,a5,a0
1c0097e4:	01076633          	or	a2,a4,a6
1c0097e8:	0a0f1463          	bnez	t5,1c009890 <__adddf3+0x662>
1c0097ec:	060f9063          	bnez	t6,1c00984c <__adddf3+0x61e>
1c0097f0:	1c061763          	bnez	a2,1c0099be <__adddf3+0x790>
1c0097f4:	ffee8593          	addi	a1,t4,-2
1c0097f8:	0015b593          	seqz	a1,a1
1c0097fc:	4701                	li	a4,0
1c0097fe:	4501                	li	a0,0
1c009800:	00e567b3          	or	a5,a0,a4
1c009804:	22078263          	beqz	a5,1c009a28 <__adddf3+0x7fa>
1c009808:	01f55793          	srli	a5,a0,0x1f
1c00980c:	00171e13          	slli	t3,a4,0x1
1c009810:	9e3e                	add	t3,t3,a5
1c009812:	00151793          	slli	a5,a0,0x1
1c009816:	f837b6b3          	p.bclr	a3,a5,28,3
1c00981a:	4881                	li	a7,0
1c00981c:	ce99                	beqz	a3,1c00983a <__adddf3+0x60c>
1c00981e:	1c2ea063          	p.beqimm	t4,2,1c0099de <__adddf3+0x7b0>
1c009822:	1a3ea563          	p.beqimm	t4,3,1c0099cc <__adddf3+0x79e>
1c009826:	4885                	li	a7,1
1c009828:	000e9963          	bnez	t4,1c00983a <__adddf3+0x60c>
1c00982c:	f647b6b3          	p.bclr	a3,a5,27,4
1c009830:	4885                	li	a7,1
1c009832:	ffc7b793          	sltiu	a5,a5,-4
1c009836:	1a46b063          	p.bneimm	a3,4,1c0099d6 <__adddf3+0x7a8>
1c00983a:	018e5e13          	srli	t3,t3,0x18
1c00983e:	001e4e13          	xori	t3,t3,1
1c009842:	fc1e3e33          	p.bclr	t3,t3,30,1
1c009846:	87ba                	mv	a5,a4
1c009848:	4301                	li	t1,0
1c00984a:	b1bd                	j	1c0094b8 <__adddf3+0x28a>
1c00984c:	16060b63          	beqz	a2,1c0099c2 <__adddf3+0x794>
1c009850:	410508b3          	sub	a7,a0,a6
1c009854:	01153333          	sltu	t1,a0,a7
1c009858:	40e78633          	sub	a2,a5,a4
1c00985c:	40660633          	sub	a2,a2,t1
1c009860:	00861313          	slli	t1,a2,0x8
1c009864:	00035c63          	bgez	t1,1c00987c <__adddf3+0x64e>
1c009868:	40a80533          	sub	a0,a6,a0
1c00986c:	40f707b3          	sub	a5,a4,a5
1c009870:	00a83733          	sltu	a4,a6,a0
1c009874:	40e78733          	sub	a4,a5,a4
1c009878:	85b6                	mv	a1,a3
1c00987a:	b759                	j	1c009800 <__adddf3+0x5d2>
1c00987c:	00c8e533          	or	a0,a7,a2
1c009880:	14051363          	bnez	a0,1c0099c6 <__adddf3+0x798>
1c009884:	ffee8593          	addi	a1,t4,-2
1c009888:	0015b593          	seqz	a1,a1
1c00988c:	4701                	li	a4,0
1c00988e:	bf8d                	j	1c009800 <__adddf3+0x5d2>
1c009890:	7ff00293          	li	t0,2047
1c009894:	025f1d63          	bne	t5,t0,1c0098ce <__adddf3+0x6a0>
1c009898:	1c0f8463          	beqz	t6,1c009a60 <__adddf3+0x832>
1c00989c:	004008b7          	lui	a7,0x400
1c0098a0:	0117f8b3          	and	a7,a5,a7
1c0098a4:	0018b893          	seqz	a7,a7
1c0098a8:	0892                	slli	a7,a7,0x4
1c0098aa:	03e31763          	bne	t1,t5,1c0098d8 <__adddf3+0x6aa>
1c0098ae:	c611                	beqz	a2,1c0098ba <__adddf3+0x68c>
1c0098b0:	00971313          	slli	t1,a4,0x9
1c0098b4:	00034363          	bltz	t1,1c0098ba <__adddf3+0x68c>
1c0098b8:	48c1                	li	a7,16
1c0098ba:	000f9f63          	bnez	t6,1c0098d8 <__adddf3+0x6aa>
1c0098be:	14060d63          	beqz	a2,1c009a18 <__adddf3+0x7ea>
1c0098c2:	87ba                	mv	a5,a4
1c0098c4:	8542                	mv	a0,a6
1c0098c6:	85b6                	mv	a1,a3
1c0098c8:	7ff00313          	li	t1,2047
1c0098cc:	b6f5                	j	1c0094b8 <__adddf3+0x28a>
1c0098ce:	4881                	li	a7,0
1c0098d0:	fe5315e3          	bne	t1,t0,1c0098ba <__adddf3+0x68c>
1c0098d4:	4881                	li	a7,0
1c0098d6:	bfe1                	j	1c0098ae <__adddf3+0x680>
1c0098d8:	da65                	beqz	a2,1c0098c8 <__adddf3+0x69a>
1c0098da:	b13d                	j	1c009508 <__adddf3+0x2da>
1c0098dc:	410508b3          	sub	a7,a0,a6
1c0098e0:	01153333          	sltu	t1,a0,a7
1c0098e4:	40e78633          	sub	a2,a5,a4
1c0098e8:	40660633          	sub	a2,a2,t1
1c0098ec:	00861313          	slli	t1,a2,0x8
1c0098f0:	06035c63          	bgez	t1,1c009968 <__adddf3+0x73a>
1c0098f4:	40a808b3          	sub	a7,a6,a0
1c0098f8:	40f707b3          	sub	a5,a4,a5
1c0098fc:	01183733          	sltu	a4,a6,a7
1c009900:	40e78633          	sub	a2,a5,a4
1c009904:	85b6                	mv	a1,a3
1c009906:	c62d                	beqz	a2,1c009970 <__adddf3+0x742>
1c009908:	100616b3          	p.fl1	a3,a2
1c00990c:	47fd                	li	a5,31
1c00990e:	40d786b3          	sub	a3,a5,a3
1c009912:	ff868313          	addi	t1,a3,-8
1c009916:	47fd                	li	a5,31
1c009918:	0667c463          	blt	a5,t1,1c009980 <__adddf3+0x752>
1c00991c:	02000793          	li	a5,32
1c009920:	406787b3          	sub	a5,a5,t1
1c009924:	00661633          	sll	a2,a2,t1
1c009928:	00f8d733          	srl	a4,a7,a5
1c00992c:	00c767b3          	or	a5,a4,a2
1c009930:	00689833          	sll	a6,a7,t1
1c009934:	09e34063          	blt	t1,t5,1c0099b4 <__adddf3+0x786>
1c009938:	41e30333          	sub	t1,t1,t5
1c00993c:	00130713          	addi	a4,t1,1
1c009940:	46fd                	li	a3,31
1c009942:	04e6c563          	blt	a3,a4,1c00998c <__adddf3+0x75e>
1c009946:	02000693          	li	a3,32
1c00994a:	8e99                	sub	a3,a3,a4
1c00994c:	00e85633          	srl	a2,a6,a4
1c009950:	00d79533          	sll	a0,a5,a3
1c009954:	00d81833          	sll	a6,a6,a3
1c009958:	8d51                	or	a0,a0,a2
1c00995a:	01003833          	snez	a6,a6
1c00995e:	01056533          	or	a0,a0,a6
1c009962:	00e7d733          	srl	a4,a5,a4
1c009966:	bd69                	j	1c009800 <__adddf3+0x5d2>
1c009968:	00c8e533          	or	a0,a7,a2
1c00996c:	fd49                	bnez	a0,1c009906 <__adddf3+0x6d8>
1c00996e:	bf19                	j	1c009884 <__adddf3+0x656>
1c009970:	100896b3          	p.fl1	a3,a7
1c009974:	47fd                	li	a5,31
1c009976:	40d786b3          	sub	a3,a5,a3
1c00997a:	02068693          	addi	a3,a3,32
1c00997e:	bf51                	j	1c009912 <__adddf3+0x6e4>
1c009980:	fd868793          	addi	a5,a3,-40
1c009984:	00f897b3          	sll	a5,a7,a5
1c009988:	4801                	li	a6,0
1c00998a:	b76d                	j	1c009934 <__adddf3+0x706>
1c00998c:	1305                	addi	t1,t1,-31
1c00998e:	02000693          	li	a3,32
1c009992:	0067d333          	srl	t1,a5,t1
1c009996:	4601                	li	a2,0
1c009998:	00d70763          	beq	a4,a3,1c0099a6 <__adddf3+0x778>
1c00999c:	04000613          	li	a2,64
1c0099a0:	8e19                	sub	a2,a2,a4
1c0099a2:	00c79633          	sll	a2,a5,a2
1c0099a6:	00c86533          	or	a0,a6,a2
1c0099aa:	00a03533          	snez	a0,a0
1c0099ae:	00a36533          	or	a0,t1,a0
1c0099b2:	bde9                	j	1c00988c <__adddf3+0x65e>
1c0099b4:	406f0333          	sub	t1,t5,t1
1c0099b8:	c177b733          	p.bclr	a4,a5,0,23
1c0099bc:	b2e5                	j	1c0093a4 <__adddf3+0x176>
1c0099be:	8542                	mv	a0,a6
1c0099c0:	bd65                	j	1c009878 <__adddf3+0x64a>
1c0099c2:	873e                	mv	a4,a5
1c0099c4:	bd35                	j	1c009800 <__adddf3+0x5d2>
1c0099c6:	8732                	mv	a4,a2
1c0099c8:	8546                	mv	a0,a7
1c0099ca:	bd1d                	j	1c009800 <__adddf3+0x5d2>
1c0099cc:	4885                	li	a7,1
1c0099ce:	e60596e3          	bnez	a1,1c00983a <__adddf3+0x60c>
1c0099d2:	ff87b793          	sltiu	a5,a5,-8
1c0099d6:	0017c793          	xori	a5,a5,1
1c0099da:	9e3e                	add	t3,t3,a5
1c0099dc:	bdb9                	j	1c00983a <__adddf3+0x60c>
1c0099de:	4885                	li	a7,1
1c0099e0:	e4058de3          	beqz	a1,1c00983a <__adddf3+0x60c>
1c0099e4:	b7fd                	j	1c0099d2 <__adddf3+0x7a4>
1c0099e6:	87ba                	mv	a5,a4
1c0099e8:	8542                	mv	a0,a6
1c0099ea:	bdf9                	j	1c0098c8 <__adddf3+0x69a>
1c0099ec:	7ff00313          	li	t1,2047
1c0099f0:	4e01                	li	t3,0
1c0099f2:	b965                	j	1c0096aa <__adddf3+0x47c>
1c0099f4:	4781                	li	a5,0
1c0099f6:	4501                	li	a0,0
1c0099f8:	7ff00313          	li	t1,2047
1c0099fc:	4881                	li	a7,0
1c0099fe:	be19                	j	1c009514 <__adddf3+0x2e6>
1c009a00:	4781                	li	a5,0
1c009a02:	bfed                	j	1c0099fc <__adddf3+0x7ce>
1c009a04:	87ba                	mv	a5,a4
1c009a06:	8542                	mv	a0,a6
1c009a08:	b7e5                	j	1c0099f0 <__adddf3+0x7c2>
1c009a0a:	4781                	li	a5,0
1c009a0c:	85b6                	mv	a1,a3
1c009a0e:	b7fd                	j	1c0099fc <__adddf3+0x7ce>
1c009a10:	87ba                	mv	a5,a4
1c009a12:	8542                	mv	a0,a6
1c009a14:	85b6                	mv	a1,a3
1c009a16:	bfe9                	j	1c0099f0 <__adddf3+0x7c2>
1c009a18:	4501                	li	a0,0
1c009a1a:	004007b7          	lui	a5,0x400
1c009a1e:	7ff00313          	li	t1,2047
1c009a22:	4581                	li	a1,0
1c009a24:	48c1                	li	a7,16
1c009a26:	b4fd                	j	1c009514 <__adddf3+0x2e6>
1c009a28:	4501                	li	a0,0
1c009a2a:	4301                	li	t1,0
1c009a2c:	bfc1                	j	1c0099fc <__adddf3+0x7ce>
1c009a2e:	a80598e3          	bnez	a1,1c0094be <__adddf3+0x290>
1c009a32:	00850713          	addi	a4,a0,8
1c009a36:	b679                	j	1c0095c4 <__adddf3+0x396>
1c009a38:	a80583e3          	beqz	a1,1c0094be <__adddf3+0x290>
1c009a3c:	bfdd                	j	1c009a32 <__adddf3+0x804>
1c009a3e:	4501                	li	a0,0
1c009a40:	000e8c63          	beqz	t4,1c009a58 <__adddf3+0x82a>
1c009a44:	003eb763          	p.bneimm	t4,3,1c009a52 <__adddf3+0x824>
1c009a48:	c981                	beqz	a1,1c009a58 <__adddf3+0x82a>
1c009a4a:	557d                	li	a0,-1
1c009a4c:	7fe00313          	li	t1,2046
1c009a50:	a021                	j	1c009a58 <__adddf3+0x82a>
1c009a52:	fe2ebce3          	p.bneimm	t4,2,1c009a4a <__adddf3+0x81c>
1c009a56:	d9f5                	beqz	a1,1c009a4a <__adddf3+0x81c>
1c009a58:	0058e893          	ori	a7,a7,5
1c009a5c:	87aa                	mv	a5,a0
1c009a5e:	b4f1                	j	1c00952a <__adddf3+0x2fc>
1c009a60:	4881                	li	a7,0
1c009a62:	e5e31ee3          	bne	t1,t5,1c0098be <__adddf3+0x690>
1c009a66:	b5bd                	j	1c0098d4 <__adddf3+0x6a6>
1c009a68:	b1e302e3          	beq	t1,t5,1c00956c <__adddf3+0x33e>
1c009a6c:	87ba                	mv	a5,a4
1c009a6e:	8542                	mv	a0,a6
1c009a70:	7ff00313          	li	t1,2047
1c009a74:	b015                	j	1c009298 <__adddf3+0x6a>
1c009a76:	8082                	ret

1c009a78 <__divdf3>:
1c009a78:	1141                	addi	sp,sp,-16
1c009a7a:	c622                	sw	s0,12(sp)
1c009a7c:	c426                	sw	s1,8(sp)
1c009a7e:	c24a                	sw	s2,4(sp)
1c009a80:	c04e                	sw	s3,0(sp)
1c009a82:	00202373          	csrr	t1,0x2
1c009a86:	d5459e33          	p.extractu	t3,a1,10,20
1c009a8a:	e6059fb3          	p.extractu	t6,a1,19,0
1c009a8e:	81fd                	srli	a1,a1,0x1f
1c009a90:	020e0463          	beqz	t3,1c009ab8 <__divdf3+0x40>
1c009a94:	7ff00793          	li	a5,2047
1c009a98:	06fe0f63          	beq	t3,a5,1c009b16 <__divdf3+0x9e>
1c009a9c:	0f8e                	slli	t6,t6,0x3
1c009a9e:	c17fcfb3          	p.bset	t6,t6,0,23
1c009aa2:	01d55793          	srli	a5,a0,0x1d
1c009aa6:	01f7efb3          	or	t6,a5,t6
1c009aaa:	00351813          	slli	a6,a0,0x3
1c009aae:	c01e0e93          	addi	t4,t3,-1023
1c009ab2:	4281                	li	t0,0
1c009ab4:	4881                	li	a7,0
1c009ab6:	a8a5                	j	1c009b2e <__divdf3+0xb6>
1c009ab8:	00afe7b3          	or	a5,t6,a0
1c009abc:	c7e1                	beqz	a5,1c009b84 <__divdf3+0x10c>
1c009abe:	020f8e63          	beqz	t6,1c009afa <__divdf3+0x82>
1c009ac2:	100f9733          	p.fl1	a4,t6
1c009ac6:	47fd                	li	a5,31
1c009ac8:	40e78733          	sub	a4,a5,a4
1c009acc:	ff570e13          	addi	t3,a4,-11 # fff5 <_l1_preload_size+0xf675>
1c009ad0:	47f1                	li	a5,28
1c009ad2:	03c7cc63          	blt	a5,t3,1c009b0a <__divdf3+0x92>
1c009ad6:	48f5                	li	a7,29
1c009ad8:	ff870813          	addi	a6,a4,-8
1c009adc:	41c888b3          	sub	a7,a7,t3
1c009ae0:	010f9fb3          	sll	t6,t6,a6
1c009ae4:	011558b3          	srl	a7,a0,a7
1c009ae8:	01f8efb3          	or	t6,a7,t6
1c009aec:	01051833          	sll	a6,a0,a6
1c009af0:	c0d00e13          	li	t3,-1011
1c009af4:	40ee0eb3          	sub	t4,t3,a4
1c009af8:	bf6d                	j	1c009ab2 <__divdf3+0x3a>
1c009afa:	10051eb3          	p.fl1	t4,a0
1c009afe:	47fd                	li	a5,31
1c009b00:	41d78eb3          	sub	t4,a5,t4
1c009b04:	020e8713          	addi	a4,t4,32
1c009b08:	b7d1                	j	1c009acc <__divdf3+0x54>
1c009b0a:	fd870f93          	addi	t6,a4,-40
1c009b0e:	01f51fb3          	sll	t6,a0,t6
1c009b12:	4801                	li	a6,0
1c009b14:	bff1                	j	1c009af0 <__divdf3+0x78>
1c009b16:	882a                	mv	a6,a0
1c009b18:	00afe533          	or	a0,t6,a0
1c009b1c:	c92d                	beqz	a0,1c009b8e <__divdf3+0x116>
1c009b1e:	00cf9793          	slli	a5,t6,0xc
1c009b22:	7ff00e93          	li	t4,2047
1c009b26:	428d                	li	t0,3
1c009b28:	48c1                	li	a7,16
1c009b2a:	f807c5e3          	bltz	a5,1c009ab4 <__divdf3+0x3c>
1c009b2e:	d5469e33          	p.extractu	t3,a3,10,20
1c009b32:	e6069733          	p.extractu	a4,a3,19,0
1c009b36:	87b2                	mv	a5,a2
1c009b38:	82fd                	srli	a3,a3,0x1f
1c009b3a:	060e0063          	beqz	t3,1c009b9a <__divdf3+0x122>
1c009b3e:	7ff00513          	li	a0,2047
1c009b42:	0aae0963          	beq	t3,a0,1c009bf4 <__divdf3+0x17c>
1c009b46:	070e                	slli	a4,a4,0x3
1c009b48:	01d65793          	srli	a5,a2,0x1d
1c009b4c:	c1774733          	p.bset	a4,a4,0,23
1c009b50:	8f5d                	or	a4,a4,a5
1c009b52:	00361793          	slli	a5,a2,0x3
1c009b56:	c01e0613          	addi	a2,t3,-1023
1c009b5a:	4f01                	li	t5,0
1c009b5c:	40ce8e33          	sub	t3,t4,a2
1c009b60:	00229613          	slli	a2,t0,0x2
1c009b64:	01e66633          	or	a2,a2,t5
1c009b68:	167d                	addi	a2,a2,-1
1c009b6a:	4eb9                	li	t4,14
1c009b6c:	00d5c533          	xor	a0,a1,a3
1c009b70:	0aceec63          	bltu	t4,a2,1c009c28 <__divdf3+0x1b0>
1c009b74:	1c001eb7          	lui	t4,0x1c001
1c009b78:	060a                	slli	a2,a2,0x2
1c009b7a:	c38e8e93          	addi	t4,t4,-968 # 1c000c38 <PIo2+0x2c>
1c009b7e:	20cef603          	p.lw	a2,a2(t4)
1c009b82:	8602                	jr	a2
1c009b84:	4f81                	li	t6,0
1c009b86:	4801                	li	a6,0
1c009b88:	4e81                	li	t4,0
1c009b8a:	4285                	li	t0,1
1c009b8c:	b725                	j	1c009ab4 <__divdf3+0x3c>
1c009b8e:	4f81                	li	t6,0
1c009b90:	4801                	li	a6,0
1c009b92:	7ff00e93          	li	t4,2047
1c009b96:	4289                	li	t0,2
1c009b98:	bf31                	j	1c009ab4 <__divdf3+0x3c>
1c009b9a:	00c767b3          	or	a5,a4,a2
1c009b9e:	c7b5                	beqz	a5,1c009c0a <__divdf3+0x192>
1c009ba0:	cf05                	beqz	a4,1c009bd8 <__divdf3+0x160>
1c009ba2:	10071e33          	p.fl1	t3,a4
1c009ba6:	47fd                	li	a5,31
1c009ba8:	41c78e33          	sub	t3,a5,t3
1c009bac:	ff5e0f13          	addi	t5,t3,-11
1c009bb0:	47f1                	li	a5,28
1c009bb2:	03e7cb63          	blt	a5,t5,1c009be8 <__divdf3+0x170>
1c009bb6:	4575                	li	a0,29
1c009bb8:	ff8e0793          	addi	a5,t3,-8
1c009bbc:	41e50533          	sub	a0,a0,t5
1c009bc0:	00f71733          	sll	a4,a4,a5
1c009bc4:	00a65533          	srl	a0,a2,a0
1c009bc8:	8f49                	or	a4,a4,a0
1c009bca:	00f617b3          	sll	a5,a2,a5
1c009bce:	c0d00613          	li	a2,-1011
1c009bd2:	41c60633          	sub	a2,a2,t3
1c009bd6:	b751                	j	1c009b5a <__divdf3+0xe2>
1c009bd8:	10061e33          	p.fl1	t3,a2
1c009bdc:	47fd                	li	a5,31
1c009bde:	41c78e33          	sub	t3,a5,t3
1c009be2:	020e0e13          	addi	t3,t3,32
1c009be6:	b7d9                	j	1c009bac <__divdf3+0x134>
1c009be8:	fd8e0713          	addi	a4,t3,-40
1c009bec:	00e61733          	sll	a4,a2,a4
1c009bf0:	4781                	li	a5,0
1c009bf2:	bff1                	j	1c009bce <__divdf3+0x156>
1c009bf4:	8e59                	or	a2,a2,a4
1c009bf6:	ce19                	beqz	a2,1c009c14 <__divdf3+0x19c>
1c009bf8:	00c71613          	slli	a2,a4,0xc
1c009bfc:	02064263          	bltz	a2,1c009c20 <__divdf3+0x1a8>
1c009c00:	7ff00613          	li	a2,2047
1c009c04:	4f0d                	li	t5,3
1c009c06:	48c1                	li	a7,16
1c009c08:	bf91                	j	1c009b5c <__divdf3+0xe4>
1c009c0a:	4701                	li	a4,0
1c009c0c:	4781                	li	a5,0
1c009c0e:	4601                	li	a2,0
1c009c10:	4f05                	li	t5,1
1c009c12:	b7a9                	j	1c009b5c <__divdf3+0xe4>
1c009c14:	4701                	li	a4,0
1c009c16:	4781                	li	a5,0
1c009c18:	7ff00613          	li	a2,2047
1c009c1c:	4f09                	li	t5,2
1c009c1e:	bf3d                	j	1c009b5c <__divdf3+0xe4>
1c009c20:	7ff00613          	li	a2,2047
1c009c24:	4f0d                	li	t5,3
1c009c26:	bf1d                	j	1c009b5c <__divdf3+0xe4>
1c009c28:	01f76663          	bltu	a4,t6,1c009c34 <__divdf3+0x1bc>
1c009c2c:	24ef9963          	bne	t6,a4,1c009e7e <__divdf3+0x406>
1c009c30:	24f86763          	bltu	a6,a5,1c009e7e <__divdf3+0x406>
1c009c34:	01ff9613          	slli	a2,t6,0x1f
1c009c38:	00185693          	srli	a3,a6,0x1
1c009c3c:	01f81393          	slli	t2,a6,0x1f
1c009c40:	001fdf93          	srli	t6,t6,0x1
1c009c44:	00d66833          	or	a6,a2,a3
1c009c48:	0722                	slli	a4,a4,0x8
1c009c4a:	0187d693          	srli	a3,a5,0x18
1c009c4e:	8ed9                	or	a3,a3,a4
1c009c50:	0106df13          	srli	t5,a3,0x10
1c009c54:	03efd2b3          	divu	t0,t6,t5
1c009c58:	1006deb3          	p.exthz	t4,a3
1c009c5c:	00879613          	slli	a2,a5,0x8
1c009c60:	01085793          	srli	a5,a6,0x10
1c009c64:	03efffb3          	remu	t6,t6,t5
1c009c68:	8716                	mv	a4,t0
1c009c6a:	025e85b3          	mul	a1,t4,t0
1c009c6e:	0fc2                	slli	t6,t6,0x10
1c009c70:	01f7e7b3          	or	a5,a5,t6
1c009c74:	00b7fc63          	bleu	a1,a5,1c009c8c <__divdf3+0x214>
1c009c78:	97b6                	add	a5,a5,a3
1c009c7a:	fff28713          	addi	a4,t0,-1
1c009c7e:	00d7e763          	bltu	a5,a3,1c009c8c <__divdf3+0x214>
1c009c82:	00b7f563          	bleu	a1,a5,1c009c8c <__divdf3+0x214>
1c009c86:	ffe28713          	addi	a4,t0,-2
1c009c8a:	97b6                	add	a5,a5,a3
1c009c8c:	8f8d                	sub	a5,a5,a1
1c009c8e:	03e7d2b3          	divu	t0,a5,t5
1c009c92:	03e7f7b3          	remu	a5,a5,t5
1c009c96:	8596                	mv	a1,t0
1c009c98:	025e8fb3          	mul	t6,t4,t0
1c009c9c:	df07a833          	p.insert	a6,a5,15,16
1c009ca0:	01f87c63          	bleu	t6,a6,1c009cb8 <__divdf3+0x240>
1c009ca4:	9836                	add	a6,a6,a3
1c009ca6:	fff28593          	addi	a1,t0,-1
1c009caa:	00d86763          	bltu	a6,a3,1c009cb8 <__divdf3+0x240>
1c009cae:	01f87563          	bleu	t6,a6,1c009cb8 <__divdf3+0x240>
1c009cb2:	ffe28593          	addi	a1,t0,-2
1c009cb6:	9836                	add	a6,a6,a3
1c009cb8:	01071793          	slli	a5,a4,0x10
1c009cbc:	64c1                	lui	s1,0x10
1c009cbe:	8fcd                	or	a5,a5,a1
1c009cc0:	41f80833          	sub	a6,a6,t6
1c009cc4:	fff48f93          	addi	t6,s1,-1 # ffff <_l1_preload_size+0xf67f>
1c009cc8:	01f7f733          	and	a4,a5,t6
1c009ccc:	0107d413          	srli	s0,a5,0x10
1c009cd0:	01f67fb3          	and	t6,a2,t6
1c009cd4:	03f405b3          	mul	a1,s0,t6
1c009cd8:	01065293          	srli	t0,a2,0x10
1c009cdc:	03f70933          	mul	s2,a4,t6
1c009ce0:	89ae                	mv	s3,a1
1c009ce2:	42e289b3          	p.mac	s3,t0,a4
1c009ce6:	01095713          	srli	a4,s2,0x10
1c009cea:	974e                	add	a4,a4,s3
1c009cec:	02540433          	mul	s0,s0,t0
1c009cf0:	00b77363          	bleu	a1,a4,1c009cf6 <__divdf3+0x27e>
1c009cf4:	9426                	add	s0,s0,s1
1c009cf6:	64c1                	lui	s1,0x10
1c009cf8:	01075593          	srli	a1,a4,0x10
1c009cfc:	14fd                	addi	s1,s1,-1
1c009cfe:	942e                	add	s0,s0,a1
1c009d00:	009775b3          	and	a1,a4,s1
1c009d04:	05c2                	slli	a1,a1,0x10
1c009d06:	00997733          	and	a4,s2,s1
1c009d0a:	95ba                	add	a1,a1,a4
1c009d0c:	00886763          	bltu	a6,s0,1c009d1a <__divdf3+0x2a2>
1c009d10:	873e                	mv	a4,a5
1c009d12:	02881e63          	bne	a6,s0,1c009d4e <__divdf3+0x2d6>
1c009d16:	02b3fc63          	bleu	a1,t2,1c009d4e <__divdf3+0x2d6>
1c009d1a:	93b2                	add	t2,t2,a2
1c009d1c:	00c3b4b3          	sltu	s1,t2,a2
1c009d20:	94b6                	add	s1,s1,a3
1c009d22:	9826                	add	a6,a6,s1
1c009d24:	fff78713          	addi	a4,a5,-1 # 3fffff <__l1_heap_size+0x300983>
1c009d28:	0106e663          	bltu	a3,a6,1c009d34 <__divdf3+0x2bc>
1c009d2c:	03069163          	bne	a3,a6,1c009d4e <__divdf3+0x2d6>
1c009d30:	00c3ef63          	bltu	t2,a2,1c009d4e <__divdf3+0x2d6>
1c009d34:	00886663          	bltu	a6,s0,1c009d40 <__divdf3+0x2c8>
1c009d38:	01041b63          	bne	s0,a6,1c009d4e <__divdf3+0x2d6>
1c009d3c:	00b3f963          	bleu	a1,t2,1c009d4e <__divdf3+0x2d6>
1c009d40:	93b2                	add	t2,t2,a2
1c009d42:	ffe78713          	addi	a4,a5,-2
1c009d46:	00c3b7b3          	sltu	a5,t2,a2
1c009d4a:	97b6                	add	a5,a5,a3
1c009d4c:	983e                	add	a6,a6,a5
1c009d4e:	40b385b3          	sub	a1,t2,a1
1c009d52:	00b3b7b3          	sltu	a5,t2,a1
1c009d56:	40880833          	sub	a6,a6,s0
1c009d5a:	40f80833          	sub	a6,a6,a5
1c009d5e:	57fd                	li	a5,-1
1c009d60:	0f068563          	beq	a3,a6,1c009e4a <__divdf3+0x3d2>
1c009d64:	03e85433          	divu	s0,a6,t5
1c009d68:	0105d793          	srli	a5,a1,0x10
1c009d6c:	03e87833          	remu	a6,a6,t5
1c009d70:	028e83b3          	mul	t2,t4,s0
1c009d74:	0842                	slli	a6,a6,0x10
1c009d76:	0107e833          	or	a6,a5,a6
1c009d7a:	87a2                	mv	a5,s0
1c009d7c:	00787c63          	bleu	t2,a6,1c009d94 <__divdf3+0x31c>
1c009d80:	9836                	add	a6,a6,a3
1c009d82:	fff40793          	addi	a5,s0,-1
1c009d86:	00d86763          	bltu	a6,a3,1c009d94 <__divdf3+0x31c>
1c009d8a:	00787563          	bleu	t2,a6,1c009d94 <__divdf3+0x31c>
1c009d8e:	ffe40793          	addi	a5,s0,-2
1c009d92:	9836                	add	a6,a6,a3
1c009d94:	40780833          	sub	a6,a6,t2
1c009d98:	03e853b3          	divu	t2,a6,t5
1c009d9c:	03e87833          	remu	a6,a6,t5
1c009da0:	027e8eb3          	mul	t4,t4,t2
1c009da4:	df0825b3          	p.insert	a1,a6,15,16
1c009da8:	881e                	mv	a6,t2
1c009daa:	01d5fc63          	bleu	t4,a1,1c009dc2 <__divdf3+0x34a>
1c009dae:	95b6                	add	a1,a1,a3
1c009db0:	fff38813          	addi	a6,t2,-1
1c009db4:	00d5e763          	bltu	a1,a3,1c009dc2 <__divdf3+0x34a>
1c009db8:	01d5f563          	bleu	t4,a1,1c009dc2 <__divdf3+0x34a>
1c009dbc:	ffe38813          	addi	a6,t2,-2
1c009dc0:	95b6                	add	a1,a1,a3
1c009dc2:	07c2                	slli	a5,a5,0x10
1c009dc4:	0107e833          	or	a6,a5,a6
1c009dc8:	100857b3          	p.exthz	a5,a6
1c009dcc:	01085f13          	srli	t5,a6,0x10
1c009dd0:	02ff8433          	mul	s0,t6,a5
1c009dd4:	41d58eb3          	sub	t4,a1,t4
1c009dd8:	03ff0fb3          	mul	t6,t5,t6
1c009ddc:	85fe                	mv	a1,t6
1c009dde:	42f285b3          	p.mac	a1,t0,a5
1c009de2:	01045793          	srli	a5,s0,0x10
1c009de6:	97ae                	add	a5,a5,a1
1c009de8:	03e283b3          	mul	t2,t0,t5
1c009dec:	01f7f463          	bleu	t6,a5,1c009df4 <__divdf3+0x37c>
1c009df0:	65c1                	lui	a1,0x10
1c009df2:	93ae                	add	t2,t2,a1
1c009df4:	6fc1                	lui	t6,0x10
1c009df6:	1ffd                	addi	t6,t6,-1
1c009df8:	0107df13          	srli	t5,a5,0x10
1c009dfc:	01f7f5b3          	and	a1,a5,t6
1c009e00:	05c2                	slli	a1,a1,0x10
1c009e02:	01f47433          	and	s0,s0,t6
1c009e06:	9f1e                	add	t5,t5,t2
1c009e08:	95a2                	add	a1,a1,s0
1c009e0a:	01eee663          	bltu	t4,t5,1c009e16 <__divdf3+0x39e>
1c009e0e:	87c2                	mv	a5,a6
1c009e10:	03ee9b63          	bne	t4,t5,1c009e46 <__divdf3+0x3ce>
1c009e14:	c99d                	beqz	a1,1c009e4a <__divdf3+0x3d2>
1c009e16:	9eb6                	add	t4,t4,a3
1c009e18:	fff80793          	addi	a5,a6,-1
1c009e1c:	02dee163          	bltu	t4,a3,1c009e3e <__divdf3+0x3c6>
1c009e20:	01eee663          	bltu	t4,t5,1c009e2c <__divdf3+0x3b4>
1c009e24:	03ee9163          	bne	t4,t5,1c009e46 <__divdf3+0x3ce>
1c009e28:	00b67d63          	bleu	a1,a2,1c009e42 <__divdf3+0x3ca>
1c009e2c:	ffe80793          	addi	a5,a6,-2
1c009e30:	00161813          	slli	a6,a2,0x1
1c009e34:	00c83633          	sltu	a2,a6,a2
1c009e38:	96b2                	add	a3,a3,a2
1c009e3a:	9eb6                	add	t4,t4,a3
1c009e3c:	8642                	mv	a2,a6
1c009e3e:	01ee9463          	bne	t4,t5,1c009e46 <__divdf3+0x3ce>
1c009e42:	00c58463          	beq	a1,a2,1c009e4a <__divdf3+0x3d2>
1c009e46:	c007c7b3          	p.bset	a5,a5,0,0
1c009e4a:	3ffe0593          	addi	a1,t3,1023
1c009e4e:	0cb05f63          	blez	a1,1c009f2c <__divdf3+0x4b4>
1c009e52:	f837b6b3          	p.bclr	a3,a5,28,3
1c009e56:	c6c1                	beqz	a3,1c009ede <__divdf3+0x466>
1c009e58:	c008c8b3          	p.bset	a7,a7,0,0
1c009e5c:	08232063          	p.beqimm	t1,2,1c009edc <__divdf3+0x464>
1c009e60:	06332a63          	p.beqimm	t1,3,1c009ed4 <__divdf3+0x45c>
1c009e64:	06031d63          	bnez	t1,1c009ede <__divdf3+0x466>
1c009e68:	f647b6b3          	p.bclr	a3,a5,27,4
1c009e6c:	0646a963          	p.beqimm	a3,4,1c009ede <__divdf3+0x466>
1c009e70:	00478693          	addi	a3,a5,4
1c009e74:	00f6b7b3          	sltu	a5,a3,a5
1c009e78:	973e                	add	a4,a4,a5
1c009e7a:	87b6                	mv	a5,a3
1c009e7c:	a08d                	j	1c009ede <__divdf3+0x466>
1c009e7e:	1e7d                	addi	t3,t3,-1
1c009e80:	4381                	li	t2,0
1c009e82:	b3d9                	j	1c009c48 <__divdf3+0x1d0>
1c009e84:	852e                	mv	a0,a1
1c009e86:	877e                	mv	a4,t6
1c009e88:	87c2                	mv	a5,a6
1c009e8a:	8f16                	mv	t5,t0
1c009e8c:	002f2d63          	p.beqimm	t5,2,1c009ea6 <__divdf3+0x42e>
1c009e90:	1c3f2763          	p.beqimm	t5,3,1c00a05e <__divdf3+0x5e6>
1c009e94:	fa1f3be3          	p.bneimm	t5,1,1c009e4a <__divdf3+0x3d2>
1c009e98:	4701                	li	a4,0
1c009e9a:	4781                	li	a5,0
1c009e9c:	aa45                	j	1c00a04c <__divdf3+0x5d4>
1c009e9e:	8536                	mv	a0,a3
1c009ea0:	b7f5                	j	1c009e8c <__divdf3+0x414>
1c009ea2:	c038c8b3          	p.bset	a7,a7,0,3
1c009ea6:	4701                	li	a4,0
1c009ea8:	4781                	li	a5,0
1c009eaa:	7ff00593          	li	a1,2047
1c009eae:	4681                	li	a3,0
1c009eb0:	e60726b3          	p.insert	a3,a4,19,0
1c009eb4:	d545a6b3          	p.insert	a3,a1,10,20
1c009eb8:	c1f526b3          	p.insert	a3,a0,0,31
1c009ebc:	85b6                	mv	a1,a3
1c009ebe:	853e                	mv	a0,a5
1c009ec0:	00088463          	beqz	a7,1c009ec8 <__divdf3+0x450>
1c009ec4:	0018a073          	csrs	0x1,a7
1c009ec8:	4432                	lw	s0,12(sp)
1c009eca:	44a2                	lw	s1,8(sp)
1c009ecc:	4912                	lw	s2,4(sp)
1c009ece:	4982                	lw	s3,0(sp)
1c009ed0:	0141                	addi	sp,sp,16
1c009ed2:	8082                	ret
1c009ed4:	e509                	bnez	a0,1c009ede <__divdf3+0x466>
1c009ed6:	00878693          	addi	a3,a5,8
1c009eda:	bf69                	j	1c009e74 <__divdf3+0x3fc>
1c009edc:	fd6d                	bnez	a0,1c009ed6 <__divdf3+0x45e>
1c009ede:	00771693          	slli	a3,a4,0x7
1c009ee2:	0006d663          	bgez	a3,1c009eee <__divdf3+0x476>
1c009ee6:	c1873733          	p.bclr	a4,a4,0,24
1c009eea:	400e0593          	addi	a1,t3,1024
1c009eee:	7fe00693          	li	a3,2046
1c009ef2:	00b6c963          	blt	a3,a1,1c009f04 <__divdf3+0x48c>
1c009ef6:	0037d693          	srli	a3,a5,0x3
1c009efa:	01d71793          	slli	a5,a4,0x1d
1c009efe:	8fd5                	or	a5,a5,a3
1c009f00:	830d                	srli	a4,a4,0x3
1c009f02:	b775                	j	1c009eae <__divdf3+0x436>
1c009f04:	02232263          	p.beqimm	t1,2,1c009f28 <__divdf3+0x4b0>
1c009f08:	00332b63          	p.beqimm	t1,3,1c009f1e <__divdf3+0x4a6>
1c009f0c:	00030a63          	beqz	t1,1c009f20 <__divdf3+0x4a8>
1c009f10:	57fd                	li	a5,-1
1c009f12:	7fe00593          	li	a1,2046
1c009f16:	0058e893          	ori	a7,a7,5
1c009f1a:	873e                	mv	a4,a5
1c009f1c:	bf49                	j	1c009eae <__divdf3+0x436>
1c009f1e:	f96d                	bnez	a0,1c009f10 <__divdf3+0x498>
1c009f20:	4781                	li	a5,0
1c009f22:	7ff00593          	li	a1,2047
1c009f26:	bfc5                	j	1c009f16 <__divdf3+0x49e>
1c009f28:	fd65                	bnez	a0,1c009f20 <__divdf3+0x4a8>
1c009f2a:	b7dd                	j	1c009f10 <__divdf3+0x498>
1c009f2c:	4805                	li	a6,1
1c009f2e:	e1a9                	bnez	a1,1c009f70 <__divdf3+0x4f8>
1c009f30:	f837b633          	p.bclr	a2,a5,28,3
1c009f34:	86ba                	mv	a3,a4
1c009f36:	ca05                	beqz	a2,1c009f66 <__divdf3+0x4ee>
1c009f38:	c008c8b3          	p.bset	a7,a7,0,0
1c009f3c:	02232463          	p.beqimm	t1,2,1c009f64 <__divdf3+0x4ec>
1c009f40:	00332e63          	p.beqimm	t1,3,1c009f5c <__divdf3+0x4e4>
1c009f44:	02031163          	bnez	t1,1c009f66 <__divdf3+0x4ee>
1c009f48:	f647b633          	p.bclr	a2,a5,27,4
1c009f4c:	00462d63          	p.beqimm	a2,4,1c009f66 <__divdf3+0x4ee>
1c009f50:	ffc7b693          	sltiu	a3,a5,-4
1c009f54:	0016c693          	xori	a3,a3,1
1c009f58:	96ba                	add	a3,a3,a4
1c009f5a:	a031                	j	1c009f66 <__divdf3+0x4ee>
1c009f5c:	e509                	bnez	a0,1c009f66 <__divdf3+0x4ee>
1c009f5e:	ff87b693          	sltiu	a3,a5,-8
1c009f62:	bfcd                	j	1c009f54 <__divdf3+0x4dc>
1c009f64:	fd6d                	bnez	a0,1c009f5e <__divdf3+0x4e6>
1c009f66:	82e1                	srli	a3,a3,0x18
1c009f68:	0016c693          	xori	a3,a3,1
1c009f6c:	fc16b833          	p.bclr	a6,a3,30,1
1c009f70:	4685                	li	a3,1
1c009f72:	8e8d                	sub	a3,a3,a1
1c009f74:	03800613          	li	a2,56
1c009f78:	0ad64b63          	blt	a2,a3,1c00a02e <__divdf3+0x5b6>
1c009f7c:	467d                	li	a2,31
1c009f7e:	04d64663          	blt	a2,a3,1c009fca <__divdf3+0x552>
1c009f82:	41ee0e13          	addi	t3,t3,1054
1c009f86:	01c71633          	sll	a2,a4,t3
1c009f8a:	00d7d5b3          	srl	a1,a5,a3
1c009f8e:	01c797b3          	sll	a5,a5,t3
1c009f92:	8e4d                	or	a2,a2,a1
1c009f94:	00f037b3          	snez	a5,a5
1c009f98:	8fd1                	or	a5,a5,a2
1c009f9a:	00d75733          	srl	a4,a4,a3
1c009f9e:	f837b6b3          	p.bclr	a3,a5,28,3
1c009fa2:	cea1                	beqz	a3,1c009ffa <__divdf3+0x582>
1c009fa4:	c008c8b3          	p.bset	a7,a7,0,0
1c009fa8:	04232863          	p.beqimm	t1,2,1c009ff8 <__divdf3+0x580>
1c009fac:	04332263          	p.beqimm	t1,3,1c009ff0 <__divdf3+0x578>
1c009fb0:	04031563          	bnez	t1,1c009ffa <__divdf3+0x582>
1c009fb4:	f647b6b3          	p.bclr	a3,a5,27,4
1c009fb8:	0446a163          	p.beqimm	a3,4,1c009ffa <__divdf3+0x582>
1c009fbc:	00478693          	addi	a3,a5,4
1c009fc0:	00f6b7b3          	sltu	a5,a3,a5
1c009fc4:	973e                	add	a4,a4,a5
1c009fc6:	87b6                	mv	a5,a3
1c009fc8:	a80d                	j	1c009ffa <__divdf3+0x582>
1c009fca:	5605                	li	a2,-31
1c009fcc:	8e0d                	sub	a2,a2,a1
1c009fce:	02000e93          	li	t4,32
1c009fd2:	00c75633          	srl	a2,a4,a2
1c009fd6:	4581                	li	a1,0
1c009fd8:	01d68663          	beq	a3,t4,1c009fe4 <__divdf3+0x56c>
1c009fdc:	43ee0e13          	addi	t3,t3,1086
1c009fe0:	01c715b3          	sll	a1,a4,t3
1c009fe4:	8fcd                	or	a5,a5,a1
1c009fe6:	00f037b3          	snez	a5,a5
1c009fea:	8fd1                	or	a5,a5,a2
1c009fec:	4701                	li	a4,0
1c009fee:	bf45                	j	1c009f9e <__divdf3+0x526>
1c009ff0:	e509                	bnez	a0,1c009ffa <__divdf3+0x582>
1c009ff2:	00878693          	addi	a3,a5,8
1c009ff6:	b7e9                	j	1c009fc0 <__divdf3+0x548>
1c009ff8:	fd6d                	bnez	a0,1c009ff2 <__divdf3+0x57a>
1c009ffa:	00871613          	slli	a2,a4,0x8
1c009ffe:	02065063          	bgez	a2,1c00a01e <__divdf3+0x5a6>
1c00a002:	c008c8b3          	p.bset	a7,a7,0,0
1c00a006:	4701                	li	a4,0
1c00a008:	4781                	li	a5,0
1c00a00a:	4585                	li	a1,1
1c00a00c:	ea0801e3          	beqz	a6,1c009eae <__divdf3+0x436>
1c00a010:	fc18b6b3          	p.bclr	a3,a7,30,1
1c00a014:	e8068de3          	beqz	a3,1c009eae <__divdf3+0x436>
1c00a018:	c018c8b3          	p.bset	a7,a7,0,1
1c00a01c:	bd49                	j	1c009eae <__divdf3+0x436>
1c00a01e:	0037d693          	srli	a3,a5,0x3
1c00a022:	01d71793          	slli	a5,a4,0x1d
1c00a026:	8fd5                	or	a5,a5,a3
1c00a028:	830d                	srli	a4,a4,0x3
1c00a02a:	4581                	li	a1,0
1c00a02c:	b7c5                	j	1c00a00c <__divdf3+0x594>
1c00a02e:	8fd9                	or	a5,a5,a4
1c00a030:	cb99                	beqz	a5,1c00a046 <__divdf3+0x5ce>
1c00a032:	c008c8b3          	p.bset	a7,a7,0,0
1c00a036:	02232163          	p.beqimm	t1,2,1c00a058 <__divdf3+0x5e0>
1c00a03a:	00332b63          	p.beqimm	t1,3,1c00a050 <__divdf3+0x5d8>
1c00a03e:	4795                	li	a5,5
1c00a040:	00031a63          	bnez	t1,1c00a054 <__divdf3+0x5dc>
1c00a044:	838d                	srli	a5,a5,0x3
1c00a046:	c018c8b3          	p.bset	a7,a7,0,1
1c00a04a:	4701                	li	a4,0
1c00a04c:	4581                	li	a1,0
1c00a04e:	b585                	j	1c009eae <__divdf3+0x436>
1c00a050:	47a5                	li	a5,9
1c00a052:	d96d                	beqz	a0,1c00a044 <__divdf3+0x5cc>
1c00a054:	4785                	li	a5,1
1c00a056:	b7fd                	j	1c00a044 <__divdf3+0x5cc>
1c00a058:	47a5                	li	a5,9
1c00a05a:	f56d                	bnez	a0,1c00a044 <__divdf3+0x5cc>
1c00a05c:	bfe5                	j	1c00a054 <__divdf3+0x5dc>
1c00a05e:	00080737          	lui	a4,0x80
1c00a062:	4781                	li	a5,0
1c00a064:	7ff00593          	li	a1,2047
1c00a068:	4501                	li	a0,0
1c00a06a:	b591                	j	1c009eae <__divdf3+0x436>
1c00a06c:	00080737          	lui	a4,0x80
1c00a070:	4781                	li	a5,0
1c00a072:	7ff00593          	li	a1,2047
1c00a076:	4501                	li	a0,0
1c00a078:	48c1                	li	a7,16
1c00a07a:	bd15                	j	1c009eae <__divdf3+0x436>

1c00a07c <__gedf2>:
1c00a07c:	002027f3          	csrr	a5,0x2
1c00a080:	7ff00e93          	li	t4,2047
1c00a084:	d5459733          	p.extractu	a4,a1,10,20
1c00a088:	e60597b3          	p.extractu	a5,a1,19,0
1c00a08c:	e60698b3          	p.extractu	a7,a3,19,0
1c00a090:	d5469833          	p.extractu	a6,a3,10,20
1c00a094:	832a                	mv	t1,a0
1c00a096:	81fd                	srli	a1,a1,0x1f
1c00a098:	8e32                	mv	t3,a2
1c00a09a:	82fd                	srli	a3,a3,0x1f
1c00a09c:	01d71a63          	bne	a4,t4,1c00a0b0 <__gedf2+0x34>
1c00a0a0:	00a7eeb3          	or	t4,a5,a0
1c00a0a4:	060e8a63          	beqz	t4,1c00a118 <__gedf2+0x9c>
1c00a0a8:	00186073          	csrsi	0x1,16
1c00a0ac:	5579                	li	a0,-2
1c00a0ae:	8082                	ret
1c00a0b0:	01d81a63          	bne	a6,t4,1c00a0c4 <__gedf2+0x48>
1c00a0b4:	00c8e633          	or	a2,a7,a2
1c00a0b8:	fa65                	bnez	a2,1c00a0a8 <__gedf2+0x2c>
1c00a0ba:	ef19                	bnez	a4,1c00a0d8 <__gedf2+0x5c>
1c00a0bc:	8d5d                	or	a0,a0,a5
1c00a0be:	00153513          	seqz	a0,a0
1c00a0c2:	a811                	j	1c00a0d6 <__gedf2+0x5a>
1c00a0c4:	ef21                	bnez	a4,1c00a11c <__gedf2+0xa0>
1c00a0c6:	8d5d                	or	a0,a0,a5
1c00a0c8:	00153513          	seqz	a0,a0
1c00a0cc:	00081563          	bnez	a6,1c00a0d6 <__gedf2+0x5a>
1c00a0d0:	00c8e633          	or	a2,a7,a2
1c00a0d4:	c221                	beqz	a2,1c00a114 <__gedf2+0x98>
1c00a0d6:	e901                	bnez	a0,1c00a0e6 <__gedf2+0x6a>
1c00a0d8:	00d59463          	bne	a1,a3,1c00a0e0 <__gedf2+0x64>
1c00a0dc:	00e85863          	ble	a4,a6,1c00a0ec <__gedf2+0x70>
1c00a0e0:	4505                	li	a0,1
1c00a0e2:	e199                	bnez	a1,1c00a0e8 <__gedf2+0x6c>
1c00a0e4:	8082                	ret
1c00a0e6:	fefd                	bnez	a3,1c00a0e4 <__gedf2+0x68>
1c00a0e8:	557d                	li	a0,-1
1c00a0ea:	8082                	ret
1c00a0ec:	01075663          	ble	a6,a4,1c00a0f8 <__gedf2+0x7c>
1c00a0f0:	557d                	li	a0,-1
1c00a0f2:	d9ed                	beqz	a1,1c00a0e4 <__gedf2+0x68>
1c00a0f4:	4505                	li	a0,1
1c00a0f6:	8082                	ret
1c00a0f8:	fef8e4e3          	bltu	a7,a5,1c00a0e0 <__gedf2+0x64>
1c00a0fc:	01179863          	bne	a5,a7,1c00a10c <__gedf2+0x90>
1c00a100:	fe6e60e3          	bltu	t3,t1,1c00a0e0 <__gedf2+0x64>
1c00a104:	4501                	li	a0,0
1c00a106:	fdc37fe3          	bleu	t3,t1,1c00a0e4 <__gedf2+0x68>
1c00a10a:	b7dd                	j	1c00a0f0 <__gedf2+0x74>
1c00a10c:	ff17e2e3          	bltu	a5,a7,1c00a0f0 <__gedf2+0x74>
1c00a110:	4501                	li	a0,0
1c00a112:	8082                	ret
1c00a114:	fd75                	bnez	a0,1c00a110 <__gedf2+0x94>
1c00a116:	b7e9                	j	1c00a0e0 <__gedf2+0x64>
1c00a118:	f8e80ee3          	beq	a6,a4,1c00a0b4 <__gedf2+0x38>
1c00a11c:	4501                	li	a0,0
1c00a11e:	fa0809e3          	beqz	a6,1c00a0d0 <__gedf2+0x54>
1c00a122:	bf5d                	j	1c00a0d8 <__gedf2+0x5c>

1c00a124 <__ledf2>:
1c00a124:	002027f3          	csrr	a5,0x2
1c00a128:	7ff00e93          	li	t4,2047
1c00a12c:	d5459733          	p.extractu	a4,a1,10,20
1c00a130:	e60597b3          	p.extractu	a5,a1,19,0
1c00a134:	e60698b3          	p.extractu	a7,a3,19,0
1c00a138:	d5469833          	p.extractu	a6,a3,10,20
1c00a13c:	832a                	mv	t1,a0
1c00a13e:	81fd                	srli	a1,a1,0x1f
1c00a140:	8e32                	mv	t3,a2
1c00a142:	82fd                	srli	a3,a3,0x1f
1c00a144:	01d71a63          	bne	a4,t4,1c00a158 <__ledf2+0x34>
1c00a148:	00a7eeb3          	or	t4,a5,a0
1c00a14c:	060e8a63          	beqz	t4,1c00a1c0 <__ledf2+0x9c>
1c00a150:	00186073          	csrsi	0x1,16
1c00a154:	4509                	li	a0,2
1c00a156:	8082                	ret
1c00a158:	01d81a63          	bne	a6,t4,1c00a16c <__ledf2+0x48>
1c00a15c:	00c8e633          	or	a2,a7,a2
1c00a160:	fa65                	bnez	a2,1c00a150 <__ledf2+0x2c>
1c00a162:	ef19                	bnez	a4,1c00a180 <__ledf2+0x5c>
1c00a164:	8d5d                	or	a0,a0,a5
1c00a166:	00153513          	seqz	a0,a0
1c00a16a:	a811                	j	1c00a17e <__ledf2+0x5a>
1c00a16c:	ef21                	bnez	a4,1c00a1c4 <__ledf2+0xa0>
1c00a16e:	8d5d                	or	a0,a0,a5
1c00a170:	00153513          	seqz	a0,a0
1c00a174:	00081563          	bnez	a6,1c00a17e <__ledf2+0x5a>
1c00a178:	00c8e633          	or	a2,a7,a2
1c00a17c:	c221                	beqz	a2,1c00a1bc <__ledf2+0x98>
1c00a17e:	e901                	bnez	a0,1c00a18e <__ledf2+0x6a>
1c00a180:	00d59463          	bne	a1,a3,1c00a188 <__ledf2+0x64>
1c00a184:	00e85863          	ble	a4,a6,1c00a194 <__ledf2+0x70>
1c00a188:	4505                	li	a0,1
1c00a18a:	e199                	bnez	a1,1c00a190 <__ledf2+0x6c>
1c00a18c:	8082                	ret
1c00a18e:	fefd                	bnez	a3,1c00a18c <__ledf2+0x68>
1c00a190:	557d                	li	a0,-1
1c00a192:	8082                	ret
1c00a194:	01075663          	ble	a6,a4,1c00a1a0 <__ledf2+0x7c>
1c00a198:	557d                	li	a0,-1
1c00a19a:	d9ed                	beqz	a1,1c00a18c <__ledf2+0x68>
1c00a19c:	4505                	li	a0,1
1c00a19e:	8082                	ret
1c00a1a0:	fef8e4e3          	bltu	a7,a5,1c00a188 <__ledf2+0x64>
1c00a1a4:	01179863          	bne	a5,a7,1c00a1b4 <__ledf2+0x90>
1c00a1a8:	fe6e60e3          	bltu	t3,t1,1c00a188 <__ledf2+0x64>
1c00a1ac:	4501                	li	a0,0
1c00a1ae:	fdc37fe3          	bleu	t3,t1,1c00a18c <__ledf2+0x68>
1c00a1b2:	b7dd                	j	1c00a198 <__ledf2+0x74>
1c00a1b4:	ff17e2e3          	bltu	a5,a7,1c00a198 <__ledf2+0x74>
1c00a1b8:	4501                	li	a0,0
1c00a1ba:	8082                	ret
1c00a1bc:	fd75                	bnez	a0,1c00a1b8 <__ledf2+0x94>
1c00a1be:	b7e9                	j	1c00a188 <__ledf2+0x64>
1c00a1c0:	f8e80ee3          	beq	a6,a4,1c00a15c <__ledf2+0x38>
1c00a1c4:	4501                	li	a0,0
1c00a1c6:	fa0809e3          	beqz	a6,1c00a178 <__ledf2+0x54>
1c00a1ca:	bf5d                	j	1c00a180 <__ledf2+0x5c>

1c00a1cc <__muldf3>:
1c00a1cc:	1101                	addi	sp,sp,-32
1c00a1ce:	ce22                	sw	s0,28(sp)
1c00a1d0:	cc26                	sw	s1,24(sp)
1c00a1d2:	ca4a                	sw	s2,20(sp)
1c00a1d4:	c84e                	sw	s3,16(sp)
1c00a1d6:	c652                	sw	s4,12(sp)
1c00a1d8:	c456                	sw	s5,8(sp)
1c00a1da:	c25a                	sw	s6,4(sp)
1c00a1dc:	00202373          	csrr	t1,0x2
1c00a1e0:	d5459e33          	p.extractu	t3,a1,10,20
1c00a1e4:	e60598b3          	p.extractu	a7,a1,19,0
1c00a1e8:	81fd                	srli	a1,a1,0x1f
1c00a1ea:	020e0463          	beqz	t3,1c00a212 <__muldf3+0x46>
1c00a1ee:	7ff00793          	li	a5,2047
1c00a1f2:	06fe0f63          	beq	t3,a5,1c00a270 <__muldf3+0xa4>
1c00a1f6:	088e                	slli	a7,a7,0x3
1c00a1f8:	c178c8b3          	p.bset	a7,a7,0,23
1c00a1fc:	01d55793          	srli	a5,a0,0x1d
1c00a200:	0117e8b3          	or	a7,a5,a7
1c00a204:	00351f93          	slli	t6,a0,0x3
1c00a208:	c01e0e13          	addi	t3,t3,-1023
1c00a20c:	4281                	li	t0,0
1c00a20e:	4801                	li	a6,0
1c00a210:	a8a5                	j	1c00a288 <__muldf3+0xbc>
1c00a212:	00a8e7b3          	or	a5,a7,a0
1c00a216:	c7e9                	beqz	a5,1c00a2e0 <__muldf3+0x114>
1c00a218:	02088e63          	beqz	a7,1c00a254 <__muldf3+0x88>
1c00a21c:	100897b3          	p.fl1	a5,a7
1c00a220:	477d                	li	a4,31
1c00a222:	40f707b3          	sub	a5,a4,a5
1c00a226:	ff578813          	addi	a6,a5,-11
1c00a22a:	4771                	li	a4,28
1c00a22c:	03074c63          	blt	a4,a6,1c00a264 <__muldf3+0x98>
1c00a230:	4775                	li	a4,29
1c00a232:	ff878f93          	addi	t6,a5,-8
1c00a236:	41070733          	sub	a4,a4,a6
1c00a23a:	01f898b3          	sll	a7,a7,t6
1c00a23e:	00e55733          	srl	a4,a0,a4
1c00a242:	011768b3          	or	a7,a4,a7
1c00a246:	01f51fb3          	sll	t6,a0,t6
1c00a24a:	c0d00e13          	li	t3,-1011
1c00a24e:	40fe0e33          	sub	t3,t3,a5
1c00a252:	bf6d                	j	1c00a20c <__muldf3+0x40>
1c00a254:	47fd                	li	a5,31
1c00a256:	10051e33          	p.fl1	t3,a0
1c00a25a:	41c78e33          	sub	t3,a5,t3
1c00a25e:	020e0793          	addi	a5,t3,32
1c00a262:	b7d1                	j	1c00a226 <__muldf3+0x5a>
1c00a264:	fd878893          	addi	a7,a5,-40
1c00a268:	011518b3          	sll	a7,a0,a7
1c00a26c:	4f81                	li	t6,0
1c00a26e:	bff1                	j	1c00a24a <__muldf3+0x7e>
1c00a270:	8faa                	mv	t6,a0
1c00a272:	00a8e533          	or	a0,a7,a0
1c00a276:	c935                	beqz	a0,1c00a2ea <__muldf3+0x11e>
1c00a278:	00c89793          	slli	a5,a7,0xc
1c00a27c:	7ff00e13          	li	t3,2047
1c00a280:	428d                	li	t0,3
1c00a282:	4841                	li	a6,16
1c00a284:	f807c5e3          	bltz	a5,1c00a20e <__muldf3+0x42>
1c00a288:	d5469533          	p.extractu	a0,a3,10,20
1c00a28c:	e6069733          	p.extractu	a4,a3,19,0
1c00a290:	87b2                	mv	a5,a2
1c00a292:	82fd                	srli	a3,a3,0x1f
1c00a294:	c12d                	beqz	a0,1c00a2f6 <__muldf3+0x12a>
1c00a296:	7ff00e93          	li	t4,2047
1c00a29a:	0bd50b63          	beq	a0,t4,1c00a350 <__muldf3+0x184>
1c00a29e:	070e                	slli	a4,a4,0x3
1c00a2a0:	01d65793          	srli	a5,a2,0x1d
1c00a2a4:	c1774733          	p.bset	a4,a4,0,23
1c00a2a8:	8f5d                	or	a4,a4,a5
1c00a2aa:	00361793          	slli	a5,a2,0x3
1c00a2ae:	c0150613          	addi	a2,a0,-1023
1c00a2b2:	4e81                	li	t4,0
1c00a2b4:	00ce0f33          	add	t5,t3,a2
1c00a2b8:	00229613          	slli	a2,t0,0x2
1c00a2bc:	01d66633          	or	a2,a2,t4
1c00a2c0:	167d                	addi	a2,a2,-1
1c00a2c2:	43b9                	li	t2,14
1c00a2c4:	00d5c533          	xor	a0,a1,a3
1c00a2c8:	001f0e13          	addi	t3,t5,1
1c00a2cc:	0ec3e963          	bltu	t2,a2,1c00a3be <__muldf3+0x1f2>
1c00a2d0:	1c001f37          	lui	t5,0x1c001
1c00a2d4:	060a                	slli	a2,a2,0x2
1c00a2d6:	c74f0f13          	addi	t5,t5,-908 # 1c000c74 <PIo2+0x68>
1c00a2da:	20cf7603          	p.lw	a2,a2(t5)
1c00a2de:	8602                	jr	a2
1c00a2e0:	4881                	li	a7,0
1c00a2e2:	4f81                	li	t6,0
1c00a2e4:	4e01                	li	t3,0
1c00a2e6:	4285                	li	t0,1
1c00a2e8:	b71d                	j	1c00a20e <__muldf3+0x42>
1c00a2ea:	4881                	li	a7,0
1c00a2ec:	4f81                	li	t6,0
1c00a2ee:	7ff00e13          	li	t3,2047
1c00a2f2:	4289                	li	t0,2
1c00a2f4:	bf29                	j	1c00a20e <__muldf3+0x42>
1c00a2f6:	00c767b3          	or	a5,a4,a2
1c00a2fa:	c7b5                	beqz	a5,1c00a366 <__muldf3+0x19a>
1c00a2fc:	cf05                	beqz	a4,1c00a334 <__muldf3+0x168>
1c00a2fe:	10071533          	p.fl1	a0,a4
1c00a302:	47fd                	li	a5,31
1c00a304:	40a78533          	sub	a0,a5,a0
1c00a308:	ff550f13          	addi	t5,a0,-11
1c00a30c:	47f1                	li	a5,28
1c00a30e:	03e7cb63          	blt	a5,t5,1c00a344 <__muldf3+0x178>
1c00a312:	4ef5                	li	t4,29
1c00a314:	ff850793          	addi	a5,a0,-8
1c00a318:	41ee8eb3          	sub	t4,t4,t5
1c00a31c:	00f71733          	sll	a4,a4,a5
1c00a320:	01d65eb3          	srl	t4,a2,t4
1c00a324:	00eee733          	or	a4,t4,a4
1c00a328:	00f617b3          	sll	a5,a2,a5
1c00a32c:	c0d00613          	li	a2,-1011
1c00a330:	8e09                	sub	a2,a2,a0
1c00a332:	b741                	j	1c00a2b2 <__muldf3+0xe6>
1c00a334:	10061533          	p.fl1	a0,a2
1c00a338:	47fd                	li	a5,31
1c00a33a:	40a78533          	sub	a0,a5,a0
1c00a33e:	02050513          	addi	a0,a0,32
1c00a342:	b7d9                	j	1c00a308 <__muldf3+0x13c>
1c00a344:	fd850713          	addi	a4,a0,-40
1c00a348:	00e61733          	sll	a4,a2,a4
1c00a34c:	4781                	li	a5,0
1c00a34e:	bff9                	j	1c00a32c <__muldf3+0x160>
1c00a350:	8e59                	or	a2,a2,a4
1c00a352:	ce19                	beqz	a2,1c00a370 <__muldf3+0x1a4>
1c00a354:	00c71613          	slli	a2,a4,0xc
1c00a358:	02064263          	bltz	a2,1c00a37c <__muldf3+0x1b0>
1c00a35c:	7ff00613          	li	a2,2047
1c00a360:	4e8d                	li	t4,3
1c00a362:	4841                	li	a6,16
1c00a364:	bf81                	j	1c00a2b4 <__muldf3+0xe8>
1c00a366:	4701                	li	a4,0
1c00a368:	4781                	li	a5,0
1c00a36a:	4601                	li	a2,0
1c00a36c:	4e85                	li	t4,1
1c00a36e:	b799                	j	1c00a2b4 <__muldf3+0xe8>
1c00a370:	4701                	li	a4,0
1c00a372:	4781                	li	a5,0
1c00a374:	7ff00613          	li	a2,2047
1c00a378:	4e89                	li	t4,2
1c00a37a:	bf2d                	j	1c00a2b4 <__muldf3+0xe8>
1c00a37c:	7ff00613          	li	a2,2047
1c00a380:	4e8d                	li	t4,3
1c00a382:	bf0d                	j	1c00a2b4 <__muldf3+0xe8>
1c00a384:	00080737          	lui	a4,0x80
1c00a388:	4781                	li	a5,0
1c00a38a:	7ff00893          	li	a7,2047
1c00a38e:	4501                	li	a0,0
1c00a390:	4841                	li	a6,16
1c00a392:	4681                	li	a3,0
1c00a394:	e60726b3          	p.insert	a3,a4,19,0
1c00a398:	d548a6b3          	p.insert	a3,a7,10,20
1c00a39c:	c1f526b3          	p.insert	a3,a0,0,31
1c00a3a0:	85b6                	mv	a1,a3
1c00a3a2:	853e                	mv	a0,a5
1c00a3a4:	00080463          	beqz	a6,1c00a3ac <__muldf3+0x1e0>
1c00a3a8:	00182073          	csrs	0x1,a6
1c00a3ac:	4472                	lw	s0,28(sp)
1c00a3ae:	44e2                	lw	s1,24(sp)
1c00a3b0:	4952                	lw	s2,20(sp)
1c00a3b2:	49c2                	lw	s3,16(sp)
1c00a3b4:	4a32                	lw	s4,12(sp)
1c00a3b6:	4aa2                	lw	s5,8(sp)
1c00a3b8:	4b12                	lw	s6,4(sp)
1c00a3ba:	6105                	addi	sp,sp,32
1c00a3bc:	8082                	ret
1c00a3be:	6641                	lui	a2,0x10
1c00a3c0:	fff60293          	addi	t0,a2,-1 # ffff <_l1_preload_size+0xf67f>
1c00a3c4:	010fd693          	srli	a3,t6,0x10
1c00a3c8:	0057f4b3          	and	s1,a5,t0
1c00a3cc:	02968433          	mul	s0,a3,s1
1c00a3d0:	005fffb3          	and	t6,t6,t0
1c00a3d4:	0107d993          	srli	s3,a5,0x10
1c00a3d8:	8ea2                	mv	t4,s0
1c00a3da:	029f87b3          	mul	a5,t6,s1
1c00a3de:	43f98eb3          	p.mac	t4,s3,t6
1c00a3e2:	0107d593          	srli	a1,a5,0x10
1c00a3e6:	95f6                	add	a1,a1,t4
1c00a3e8:	033683b3          	mul	t2,a3,s3
1c00a3ec:	0085f363          	bleu	s0,a1,1c00a3f2 <__muldf3+0x226>
1c00a3f0:	93b2                	add	t2,t2,a2
1c00a3f2:	0055feb3          	and	t4,a1,t0
1c00a3f6:	0057f7b3          	and	a5,a5,t0
1c00a3fa:	0ec2                	slli	t4,t4,0x10
1c00a3fc:	9ebe                	add	t4,t4,a5
1c00a3fe:	005777b3          	and	a5,a4,t0
1c00a402:	01075913          	srli	s2,a4,0x10
1c00a406:	0105da13          	srli	s4,a1,0x10
1c00a40a:	02f68633          	mul	a2,a3,a5
1c00a40e:	85b2                	mv	a1,a2
1c00a410:	02ff8733          	mul	a4,t6,a5
1c00a414:	43f905b3          	p.mac	a1,s2,t6
1c00a418:	8fae                	mv	t6,a1
1c00a41a:	01075593          	srli	a1,a4,0x10
1c00a41e:	95fe                	add	a1,a1,t6
1c00a420:	032686b3          	mul	a3,a3,s2
1c00a424:	00c5f463          	bleu	a2,a1,1c00a42c <__muldf3+0x260>
1c00a428:	6641                	lui	a2,0x10
1c00a42a:	96b2                	add	a3,a3,a2
1c00a42c:	6ac1                	lui	s5,0x10
1c00a42e:	fffa8413          	addi	s0,s5,-1 # ffff <_l1_preload_size+0xf67f>
1c00a432:	0085f2b3          	and	t0,a1,s0
1c00a436:	0105d613          	srli	a2,a1,0x10
1c00a43a:	008775b3          	and	a1,a4,s0
1c00a43e:	0108d713          	srli	a4,a7,0x10
1c00a442:	0088f433          	and	s0,a7,s0
1c00a446:	02940b33          	mul	s6,s0,s1
1c00a44a:	02c2                	slli	t0,t0,0x10
1c00a44c:	92ae                	add	t0,t0,a1
1c00a44e:	9636                	add	a2,a2,a3
1c00a450:	9a16                	add	s4,s4,t0
1c00a452:	029704b3          	mul	s1,a4,s1
1c00a456:	010b5f93          	srli	t6,s6,0x10
1c00a45a:	88a6                	mv	a7,s1
1c00a45c:	428988b3          	p.mac	a7,s3,s0
1c00a460:	9fc6                	add	t6,t6,a7
1c00a462:	02e986b3          	mul	a3,s3,a4
1c00a466:	009ff363          	bleu	s1,t6,1c00a46c <__muldf3+0x2a0>
1c00a46a:	96d6                	add	a3,a3,s5
1c00a46c:	010fd493          	srli	s1,t6,0x10
1c00a470:	65c1                	lui	a1,0x10
1c00a472:	94b6                	add	s1,s1,a3
1c00a474:	fff58693          	addi	a3,a1,-1 # ffff <_l1_preload_size+0xf67f>
1c00a478:	00dff8b3          	and	a7,t6,a3
1c00a47c:	00db7fb3          	and	t6,s6,a3
1c00a480:	08c2                	slli	a7,a7,0x10
1c00a482:	028786b3          	mul	a3,a5,s0
1c00a486:	98fe                	add	a7,a7,t6
1c00a488:	02f707b3          	mul	a5,a4,a5
1c00a48c:	02e90fb3          	mul	t6,s2,a4
1c00a490:	873e                	mv	a4,a5
1c00a492:	42890733          	p.mac	a4,s2,s0
1c00a496:	0106d413          	srli	s0,a3,0x10
1c00a49a:	9722                	add	a4,a4,s0
1c00a49c:	00f77363          	bleu	a5,a4,1c00a4a2 <__muldf3+0x2d6>
1c00a4a0:	9fae                	add	t6,t6,a1
1c00a4a2:	67c1                	lui	a5,0x10
1c00a4a4:	17fd                	addi	a5,a5,-1
1c00a4a6:	00f775b3          	and	a1,a4,a5
1c00a4aa:	8efd                	and	a3,a3,a5
1c00a4ac:	05c2                	slli	a1,a1,0x10
1c00a4ae:	93d2                	add	t2,t2,s4
1c00a4b0:	95b6                	add	a1,a1,a3
1c00a4b2:	00c586b3          	add	a3,a1,a2
1c00a4b6:	0053b2b3          	sltu	t0,t2,t0
1c00a4ba:	00568933          	add	s2,a3,t0
1c00a4be:	00c6b6b3          	sltu	a3,a3,a2
1c00a4c2:	8636                	mv	a2,a3
1c00a4c4:	005936b3          	sltu	a3,s2,t0
1c00a4c8:	93c6                	add	t2,t2,a7
1c00a4ca:	8ed1                	or	a3,a3,a2
1c00a4cc:	0113b8b3          	sltu	a7,t2,a7
1c00a4d0:	009905b3          	add	a1,s2,s1
1c00a4d4:	8341                	srli	a4,a4,0x10
1c00a4d6:	0ff6f693          	andi	a3,a3,255
1c00a4da:	01158433          	add	s0,a1,a7
1c00a4de:	96ba                	add	a3,a3,a4
1c00a4e0:	0095b733          	sltu	a4,a1,s1
1c00a4e4:	85ba                	mv	a1,a4
1c00a4e6:	01143733          	sltu	a4,s0,a7
1c00a4ea:	8f4d                	or	a4,a4,a1
1c00a4ec:	0ff77713          	andi	a4,a4,255
1c00a4f0:	9736                	add	a4,a4,a3
1c00a4f2:	977e                	add	a4,a4,t6
1c00a4f4:	01745793          	srli	a5,s0,0x17
1c00a4f8:	0726                	slli	a4,a4,0x9
1c00a4fa:	8f5d                	or	a4,a4,a5
1c00a4fc:	00939793          	slli	a5,t2,0x9
1c00a500:	01d7e7b3          	or	a5,a5,t4
1c00a504:	00f037b3          	snez	a5,a5
1c00a508:	0173d393          	srli	t2,t2,0x17
1c00a50c:	0077e7b3          	or	a5,a5,t2
1c00a510:	0426                	slli	s0,s0,0x9
1c00a512:	00771613          	slli	a2,a4,0x7
1c00a516:	8fc1                	or	a5,a5,s0
1c00a518:	06065963          	bgez	a2,1c00a58a <__muldf3+0x3be>
1c00a51c:	0017d693          	srli	a3,a5,0x1
1c00a520:	fc17b7b3          	p.bclr	a5,a5,30,1
1c00a524:	8fd5                	or	a5,a5,a3
1c00a526:	01f71693          	slli	a3,a4,0x1f
1c00a52a:	8fd5                	or	a5,a5,a3
1c00a52c:	8305                	srli	a4,a4,0x1
1c00a52e:	3ffe0893          	addi	a7,t3,1023
1c00a532:	0b105a63          	blez	a7,1c00a5e6 <__muldf3+0x41a>
1c00a536:	f837b6b3          	p.bclr	a3,a5,28,3
1c00a53a:	ceb9                	beqz	a3,1c00a598 <__muldf3+0x3cc>
1c00a53c:	c0084833          	p.bset	a6,a6,0,0
1c00a540:	04232b63          	p.beqimm	t1,2,1c00a596 <__muldf3+0x3ca>
1c00a544:	04332563          	p.beqimm	t1,3,1c00a58e <__muldf3+0x3c2>
1c00a548:	04031863          	bnez	t1,1c00a598 <__muldf3+0x3cc>
1c00a54c:	f647b6b3          	p.bclr	a3,a5,27,4
1c00a550:	0446a463          	p.beqimm	a3,4,1c00a598 <__muldf3+0x3cc>
1c00a554:	00478693          	addi	a3,a5,4 # 10004 <_l1_preload_size+0xf684>
1c00a558:	00f6b7b3          	sltu	a5,a3,a5
1c00a55c:	973e                	add	a4,a4,a5
1c00a55e:	87b6                	mv	a5,a3
1c00a560:	a825                	j	1c00a598 <__muldf3+0x3cc>
1c00a562:	852e                	mv	a0,a1
1c00a564:	8746                	mv	a4,a7
1c00a566:	87fe                	mv	a5,t6
1c00a568:	8e96                	mv	t4,t0
1c00a56a:	002eab63          	p.beqimm	t4,2,1c00a580 <__muldf3+0x3b4>
1c00a56e:	1a3eab63          	p.beqimm	t4,3,1c00a724 <__muldf3+0x558>
1c00a572:	fa1ebee3          	p.bneimm	t4,1,1c00a52e <__muldf3+0x362>
1c00a576:	4701                	li	a4,0
1c00a578:	4781                	li	a5,0
1c00a57a:	aa61                	j	1c00a712 <__muldf3+0x546>
1c00a57c:	8536                	mv	a0,a3
1c00a57e:	b7f5                	j	1c00a56a <__muldf3+0x39e>
1c00a580:	4701                	li	a4,0
1c00a582:	4781                	li	a5,0
1c00a584:	7ff00893          	li	a7,2047
1c00a588:	b529                	j	1c00a392 <__muldf3+0x1c6>
1c00a58a:	8e7a                	mv	t3,t5
1c00a58c:	b74d                	j	1c00a52e <__muldf3+0x362>
1c00a58e:	e509                	bnez	a0,1c00a598 <__muldf3+0x3cc>
1c00a590:	00878693          	addi	a3,a5,8
1c00a594:	b7d1                	j	1c00a558 <__muldf3+0x38c>
1c00a596:	fd6d                	bnez	a0,1c00a590 <__muldf3+0x3c4>
1c00a598:	00771693          	slli	a3,a4,0x7
1c00a59c:	0006d663          	bgez	a3,1c00a5a8 <__muldf3+0x3dc>
1c00a5a0:	c1873733          	p.bclr	a4,a4,0,24
1c00a5a4:	400e0893          	addi	a7,t3,1024
1c00a5a8:	7fe00693          	li	a3,2046
1c00a5ac:	0116c963          	blt	a3,a7,1c00a5be <__muldf3+0x3f2>
1c00a5b0:	0037d693          	srli	a3,a5,0x3
1c00a5b4:	01d71793          	slli	a5,a4,0x1d
1c00a5b8:	8fd5                	or	a5,a5,a3
1c00a5ba:	830d                	srli	a4,a4,0x3
1c00a5bc:	bbd9                	j	1c00a392 <__muldf3+0x1c6>
1c00a5be:	02232263          	p.beqimm	t1,2,1c00a5e2 <__muldf3+0x416>
1c00a5c2:	00332b63          	p.beqimm	t1,3,1c00a5d8 <__muldf3+0x40c>
1c00a5c6:	00030a63          	beqz	t1,1c00a5da <__muldf3+0x40e>
1c00a5ca:	57fd                	li	a5,-1
1c00a5cc:	7fe00893          	li	a7,2046
1c00a5d0:	00586813          	ori	a6,a6,5
1c00a5d4:	873e                	mv	a4,a5
1c00a5d6:	bb75                	j	1c00a392 <__muldf3+0x1c6>
1c00a5d8:	f96d                	bnez	a0,1c00a5ca <__muldf3+0x3fe>
1c00a5da:	4781                	li	a5,0
1c00a5dc:	7ff00893          	li	a7,2047
1c00a5e0:	bfc5                	j	1c00a5d0 <__muldf3+0x404>
1c00a5e2:	fd65                	bnez	a0,1c00a5da <__muldf3+0x40e>
1c00a5e4:	b7dd                	j	1c00a5ca <__muldf3+0x3fe>
1c00a5e6:	4585                	li	a1,1
1c00a5e8:	04089363          	bnez	a7,1c00a62e <__muldf3+0x462>
1c00a5ec:	f837b6b3          	p.bclr	a3,a5,28,3
1c00a5f0:	863a                	mv	a2,a4
1c00a5f2:	ca85                	beqz	a3,1c00a622 <__muldf3+0x456>
1c00a5f4:	c0084833          	p.bset	a6,a6,0,0
1c00a5f8:	02232463          	p.beqimm	t1,2,1c00a620 <__muldf3+0x454>
1c00a5fc:	00332e63          	p.beqimm	t1,3,1c00a618 <__muldf3+0x44c>
1c00a600:	02031163          	bnez	t1,1c00a622 <__muldf3+0x456>
1c00a604:	f647b6b3          	p.bclr	a3,a5,27,4
1c00a608:	0046ad63          	p.beqimm	a3,4,1c00a622 <__muldf3+0x456>
1c00a60c:	ffc7b613          	sltiu	a2,a5,-4
1c00a610:	00164613          	xori	a2,a2,1
1c00a614:	963a                	add	a2,a2,a4
1c00a616:	a031                	j	1c00a622 <__muldf3+0x456>
1c00a618:	e509                	bnez	a0,1c00a622 <__muldf3+0x456>
1c00a61a:	ff87b613          	sltiu	a2,a5,-8
1c00a61e:	bfcd                	j	1c00a610 <__muldf3+0x444>
1c00a620:	fd6d                	bnez	a0,1c00a61a <__muldf3+0x44e>
1c00a622:	01865593          	srli	a1,a2,0x18
1c00a626:	0015c593          	xori	a1,a1,1
1c00a62a:	fc15b5b3          	p.bclr	a1,a1,30,1
1c00a62e:	4605                	li	a2,1
1c00a630:	41160633          	sub	a2,a2,a7
1c00a634:	03800693          	li	a3,56
1c00a638:	0ac6ce63          	blt	a3,a2,1c00a6f4 <__muldf3+0x528>
1c00a63c:	46fd                	li	a3,31
1c00a63e:	04c6c763          	blt	a3,a2,1c00a68c <__muldf3+0x4c0>
1c00a642:	41ee0e13          	addi	t3,t3,1054
1c00a646:	01c716b3          	sll	a3,a4,t3
1c00a64a:	00c7d8b3          	srl	a7,a5,a2
1c00a64e:	01c797b3          	sll	a5,a5,t3
1c00a652:	0116e6b3          	or	a3,a3,a7
1c00a656:	00f037b3          	snez	a5,a5
1c00a65a:	8fd5                	or	a5,a5,a3
1c00a65c:	00c75733          	srl	a4,a4,a2
1c00a660:	f837b6b3          	p.bclr	a3,a5,28,3
1c00a664:	ceb1                	beqz	a3,1c00a6c0 <__muldf3+0x4f4>
1c00a666:	c0084833          	p.bset	a6,a6,0,0
1c00a66a:	04232a63          	p.beqimm	t1,2,1c00a6be <__muldf3+0x4f2>
1c00a66e:	04332463          	p.beqimm	t1,3,1c00a6b6 <__muldf3+0x4ea>
1c00a672:	04031763          	bnez	t1,1c00a6c0 <__muldf3+0x4f4>
1c00a676:	f647b6b3          	p.bclr	a3,a5,27,4
1c00a67a:	0446a363          	p.beqimm	a3,4,1c00a6c0 <__muldf3+0x4f4>
1c00a67e:	00478693          	addi	a3,a5,4
1c00a682:	00f6b7b3          	sltu	a5,a3,a5
1c00a686:	973e                	add	a4,a4,a5
1c00a688:	87b6                	mv	a5,a3
1c00a68a:	a81d                	j	1c00a6c0 <__muldf3+0x4f4>
1c00a68c:	5685                	li	a3,-31
1c00a68e:	411686b3          	sub	a3,a3,a7
1c00a692:	02000e93          	li	t4,32
1c00a696:	00d756b3          	srl	a3,a4,a3
1c00a69a:	4881                	li	a7,0
1c00a69c:	01d60663          	beq	a2,t4,1c00a6a8 <__muldf3+0x4dc>
1c00a6a0:	43ee0e13          	addi	t3,t3,1086
1c00a6a4:	01c718b3          	sll	a7,a4,t3
1c00a6a8:	00f8e7b3          	or	a5,a7,a5
1c00a6ac:	00f037b3          	snez	a5,a5
1c00a6b0:	8fd5                	or	a5,a5,a3
1c00a6b2:	4701                	li	a4,0
1c00a6b4:	b775                	j	1c00a660 <__muldf3+0x494>
1c00a6b6:	e509                	bnez	a0,1c00a6c0 <__muldf3+0x4f4>
1c00a6b8:	00878693          	addi	a3,a5,8
1c00a6bc:	b7d9                	j	1c00a682 <__muldf3+0x4b6>
1c00a6be:	fd6d                	bnez	a0,1c00a6b8 <__muldf3+0x4ec>
1c00a6c0:	00871613          	slli	a2,a4,0x8
1c00a6c4:	02065063          	bgez	a2,1c00a6e4 <__muldf3+0x518>
1c00a6c8:	c0084833          	p.bset	a6,a6,0,0
1c00a6cc:	4701                	li	a4,0
1c00a6ce:	4781                	li	a5,0
1c00a6d0:	4885                	li	a7,1
1c00a6d2:	cc0580e3          	beqz	a1,1c00a392 <__muldf3+0x1c6>
1c00a6d6:	fc1836b3          	p.bclr	a3,a6,30,1
1c00a6da:	ca068ce3          	beqz	a3,1c00a392 <__muldf3+0x1c6>
1c00a6de:	c0184833          	p.bset	a6,a6,0,1
1c00a6e2:	b945                	j	1c00a392 <__muldf3+0x1c6>
1c00a6e4:	0037d693          	srli	a3,a5,0x3
1c00a6e8:	01d71793          	slli	a5,a4,0x1d
1c00a6ec:	8fd5                	or	a5,a5,a3
1c00a6ee:	830d                	srli	a4,a4,0x3
1c00a6f0:	4881                	li	a7,0
1c00a6f2:	b7c5                	j	1c00a6d2 <__muldf3+0x506>
1c00a6f4:	8fd9                	or	a5,a5,a4
1c00a6f6:	cb99                	beqz	a5,1c00a70c <__muldf3+0x540>
1c00a6f8:	c0084833          	p.bset	a6,a6,0,0
1c00a6fc:	02232163          	p.beqimm	t1,2,1c00a71e <__muldf3+0x552>
1c00a700:	00332b63          	p.beqimm	t1,3,1c00a716 <__muldf3+0x54a>
1c00a704:	4795                	li	a5,5
1c00a706:	00031a63          	bnez	t1,1c00a71a <__muldf3+0x54e>
1c00a70a:	838d                	srli	a5,a5,0x3
1c00a70c:	c0184833          	p.bset	a6,a6,0,1
1c00a710:	4701                	li	a4,0
1c00a712:	4881                	li	a7,0
1c00a714:	b9bd                	j	1c00a392 <__muldf3+0x1c6>
1c00a716:	47a5                	li	a5,9
1c00a718:	d96d                	beqz	a0,1c00a70a <__muldf3+0x53e>
1c00a71a:	4785                	li	a5,1
1c00a71c:	b7fd                	j	1c00a70a <__muldf3+0x53e>
1c00a71e:	47a5                	li	a5,9
1c00a720:	f56d                	bnez	a0,1c00a70a <__muldf3+0x53e>
1c00a722:	bfe5                	j	1c00a71a <__muldf3+0x54e>
1c00a724:	00080737          	lui	a4,0x80
1c00a728:	4781                	li	a5,0
1c00a72a:	7ff00893          	li	a7,2047
1c00a72e:	4501                	li	a0,0
1c00a730:	b18d                	j	1c00a392 <__muldf3+0x1c6>

1c00a732 <__subdf3>:
1c00a732:	00202373          	csrr	t1,0x2
1c00a736:	e60597b3          	p.extractu	a5,a1,19,0
1c00a73a:	00379893          	slli	a7,a5,0x3
1c00a73e:	01d55793          	srli	a5,a0,0x1d
1c00a742:	0117e7b3          	or	a5,a5,a7
1c00a746:	e60698b3          	p.extractu	a7,a3,19,0
1c00a74a:	01d65713          	srli	a4,a2,0x1d
1c00a74e:	d5469f33          	p.extractu	t5,a3,10,20
1c00a752:	088e                	slli	a7,a7,0x3
1c00a754:	00361813          	slli	a6,a2,0x3
1c00a758:	7ff00613          	li	a2,2047
1c00a75c:	d5459eb3          	p.extractu	t4,a1,10,20
1c00a760:	050e                	slli	a0,a0,0x3
1c00a762:	81fd                	srli	a1,a1,0x1f
1c00a764:	82fd                	srli	a3,a3,0x1f
1c00a766:	01176733          	or	a4,a4,a7
1c00a76a:	00cf1563          	bne	t5,a2,1c00a774 <__subdf3+0x42>
1c00a76e:	01076633          	or	a2,a4,a6
1c00a772:	e219                	bnez	a2,1c00a778 <__subdf3+0x46>
1c00a774:	0016c693          	xori	a3,a3,1
1c00a778:	41ee88b3          	sub	a7,t4,t5
1c00a77c:	3ab69863          	bne	a3,a1,1c00ab2c <__subdf3+0x3fa>
1c00a780:	13105863          	blez	a7,1c00a8b0 <__subdf3+0x17e>
1c00a784:	080f1863          	bnez	t5,1c00a814 <__subdf3+0xe2>
1c00a788:	010766b3          	or	a3,a4,a6
1c00a78c:	e28d                	bnez	a3,1c00a7ae <__subdf3+0x7c>
1c00a78e:	7ff00713          	li	a4,2047
1c00a792:	3ae89863          	bne	a7,a4,1c00ab42 <__subdf3+0x410>
1c00a796:	00a7e733          	or	a4,a5,a0
1c00a79a:	78070763          	beqz	a4,1c00af28 <__subdf3+0x7f6>
1c00a79e:	00979613          	slli	a2,a5,0x9
1c00a7a2:	7ff00f13          	li	t5,2047
1c00a7a6:	4881                	li	a7,0
1c00a7a8:	06065463          	bgez	a2,1c00a810 <__subdf3+0xde>
1c00a7ac:	ae09                	j	1c00aabe <__subdf3+0x38c>
1c00a7ae:	fff88693          	addi	a3,a7,-1 # 3fffff <__l1_heap_size+0x300983>
1c00a7b2:	e2a1                	bnez	a3,1c00a7f2 <__subdf3+0xc0>
1c00a7b4:	982a                	add	a6,a6,a0
1c00a7b6:	97ba                	add	a5,a5,a4
1c00a7b8:	00a83533          	sltu	a0,a6,a0
1c00a7bc:	00a78733          	add	a4,a5,a0
1c00a7c0:	4f05                	li	t5,1
1c00a7c2:	00871613          	slli	a2,a4,0x8
1c00a7c6:	46065263          	bgez	a2,1c00ac2a <__subdf3+0x4f8>
1c00a7ca:	0f05                	addi	t5,t5,1
1c00a7cc:	7ff00793          	li	a5,2047
1c00a7d0:	32ff0c63          	beq	t5,a5,1c00ab08 <__subdf3+0x3d6>
1c00a7d4:	c17737b3          	p.bclr	a5,a4,0,23
1c00a7d8:	00185693          	srli	a3,a6,0x1
1c00a7dc:	fc183833          	p.bclr	a6,a6,30,1
1c00a7e0:	01f79513          	slli	a0,a5,0x1f
1c00a7e4:	0106e833          	or	a6,a3,a6
1c00a7e8:	01056533          	or	a0,a0,a6
1c00a7ec:	8385                	srli	a5,a5,0x1
1c00a7ee:	4881                	li	a7,0
1c00a7f0:	a4f9                	j	1c00aabe <__subdf3+0x38c>
1c00a7f2:	7ff00613          	li	a2,2047
1c00a7f6:	04c89263          	bne	a7,a2,1c00a83a <__subdf3+0x108>
1c00a7fa:	00a7e733          	or	a4,a5,a0
1c00a7fe:	72070563          	beqz	a4,1c00af28 <__subdf3+0x7f6>
1c00a802:	00979713          	slli	a4,a5,0x9
1c00a806:	7ff00f13          	li	t5,2047
1c00a80a:	4881                	li	a7,0
1c00a80c:	2a074963          	bltz	a4,1c00aabe <__subdf3+0x38c>
1c00a810:	4e41                	li	t3,16
1c00a812:	aac9                	j	1c00a9e4 <__subdf3+0x2b2>
1c00a814:	7ff00693          	li	a3,2047
1c00a818:	00de9e63          	bne	t4,a3,1c00a834 <__subdf3+0x102>
1c00a81c:	00a7e733          	or	a4,a5,a0
1c00a820:	70070463          	beqz	a4,1c00af28 <__subdf3+0x7f6>
1c00a824:	00979e13          	slli	t3,a5,0x9
1c00a828:	7ff00f13          	li	t5,2047
1c00a82c:	4881                	li	a7,0
1c00a82e:	fe0e51e3          	bgez	t3,1c00a810 <__subdf3+0xde>
1c00a832:	a471                	j	1c00aabe <__subdf3+0x38c>
1c00a834:	c1774733          	p.bset	a4,a4,0,23
1c00a838:	86c6                	mv	a3,a7
1c00a83a:	03800613          	li	a2,56
1c00a83e:	06d64463          	blt	a2,a3,1c00a8a6 <__subdf3+0x174>
1c00a842:	467d                	li	a2,31
1c00a844:	02d64c63          	blt	a2,a3,1c00a87c <__subdf3+0x14a>
1c00a848:	02000893          	li	a7,32
1c00a84c:	40d888b3          	sub	a7,a7,a3
1c00a850:	00d85e33          	srl	t3,a6,a3
1c00a854:	01171633          	sll	a2,a4,a7
1c00a858:	01181833          	sll	a6,a6,a7
1c00a85c:	01c66633          	or	a2,a2,t3
1c00a860:	01003833          	snez	a6,a6
1c00a864:	01066833          	or	a6,a2,a6
1c00a868:	00d756b3          	srl	a3,a4,a3
1c00a86c:	982a                	add	a6,a6,a0
1c00a86e:	96be                	add	a3,a3,a5
1c00a870:	00a837b3          	sltu	a5,a6,a0
1c00a874:	00f68733          	add	a4,a3,a5
1c00a878:	8f76                	mv	t5,t4
1c00a87a:	b7a1                	j	1c00a7c2 <__subdf3+0x90>
1c00a87c:	02000e13          	li	t3,32
1c00a880:	00d75633          	srl	a2,a4,a3
1c00a884:	4881                	li	a7,0
1c00a886:	01c68863          	beq	a3,t3,1c00a896 <__subdf3+0x164>
1c00a88a:	04000893          	li	a7,64
1c00a88e:	40d886b3          	sub	a3,a7,a3
1c00a892:	00d718b3          	sll	a7,a4,a3
1c00a896:	0108e833          	or	a6,a7,a6
1c00a89a:	01003833          	snez	a6,a6
1c00a89e:	01066833          	or	a6,a2,a6
1c00a8a2:	4681                	li	a3,0
1c00a8a4:	b7e1                	j	1c00a86c <__subdf3+0x13a>
1c00a8a6:	01076833          	or	a6,a4,a6
1c00a8aa:	01003833          	snez	a6,a6
1c00a8ae:	bfd5                	j	1c00a8a2 <__subdf3+0x170>
1c00a8b0:	0e088a63          	beqz	a7,1c00a9a4 <__subdf3+0x272>
1c00a8b4:	040e9e63          	bnez	t4,1c00a910 <__subdf3+0x1de>
1c00a8b8:	00a7e6b3          	or	a3,a5,a0
1c00a8bc:	e28d                	bnez	a3,1c00a8de <__subdf3+0x1ac>
1c00a8be:	7ff00793          	li	a5,2047
1c00a8c2:	36ff1463          	bne	t5,a5,1c00ac2a <__subdf3+0x4f8>
1c00a8c6:	01076533          	or	a0,a4,a6
1c00a8ca:	66050563          	beqz	a0,1c00af34 <__subdf3+0x802>
1c00a8ce:	00971e93          	slli	t4,a4,0x9
1c00a8d2:	87ba                	mv	a5,a4
1c00a8d4:	8542                	mv	a0,a6
1c00a8d6:	4881                	li	a7,0
1c00a8d8:	f20edce3          	bgez	t4,1c00a810 <__subdf3+0xde>
1c00a8dc:	a2cd                	j	1c00aabe <__subdf3+0x38c>
1c00a8de:	01f8ba63          	p.bneimm	a7,-1,1c00a8f2 <__subdf3+0x1c0>
1c00a8e2:	010506b3          	add	a3,a0,a6
1c00a8e6:	97ba                	add	a5,a5,a4
1c00a8e8:	0106b733          	sltu	a4,a3,a6
1c00a8ec:	973e                	add	a4,a4,a5
1c00a8ee:	8836                	mv	a6,a3
1c00a8f0:	bdc9                	j	1c00a7c2 <__subdf3+0x90>
1c00a8f2:	7ff00693          	li	a3,2047
1c00a8f6:	fff8c893          	not	a7,a7
1c00a8fa:	02df1e63          	bne	t5,a3,1c00a936 <__subdf3+0x204>
1c00a8fe:	01076533          	or	a0,a4,a6
1c00a902:	62050963          	beqz	a0,1c00af34 <__subdf3+0x802>
1c00a906:	00971613          	slli	a2,a4,0x9
1c00a90a:	87ba                	mv	a5,a4
1c00a90c:	8542                	mv	a0,a6
1c00a90e:	bd61                	j	1c00a7a6 <__subdf3+0x74>
1c00a910:	7ff00693          	li	a3,2047
1c00a914:	00df1d63          	bne	t5,a3,1c00a92e <__subdf3+0x1fc>
1c00a918:	01076533          	or	a0,a4,a6
1c00a91c:	60050c63          	beqz	a0,1c00af34 <__subdf3+0x802>
1c00a920:	00971793          	slli	a5,a4,0x9
1c00a924:	5e07db63          	bgez	a5,1c00af1a <__subdf3+0x7e8>
1c00a928:	87ba                	mv	a5,a4
1c00a92a:	8542                	mv	a0,a6
1c00a92c:	b5c9                	j	1c00a7ee <__subdf3+0xbc>
1c00a92e:	411008b3          	neg	a7,a7
1c00a932:	c177c7b3          	p.bset	a5,a5,0,23
1c00a936:	03800693          	li	a3,56
1c00a93a:	0716c163          	blt	a3,a7,1c00a99c <__subdf3+0x26a>
1c00a93e:	46fd                	li	a3,31
1c00a940:	0316cb63          	blt	a3,a7,1c00a976 <__subdf3+0x244>
1c00a944:	02000613          	li	a2,32
1c00a948:	41160633          	sub	a2,a2,a7
1c00a94c:	01155e33          	srl	t3,a0,a7
1c00a950:	00c796b3          	sll	a3,a5,a2
1c00a954:	00c51533          	sll	a0,a0,a2
1c00a958:	01c6e6b3          	or	a3,a3,t3
1c00a95c:	00a03533          	snez	a0,a0
1c00a960:	8d55                	or	a0,a0,a3
1c00a962:	0117d8b3          	srl	a7,a5,a7
1c00a966:	9542                	add	a0,a0,a6
1c00a968:	00e887b3          	add	a5,a7,a4
1c00a96c:	01053733          	sltu	a4,a0,a6
1c00a970:	973e                	add	a4,a4,a5
1c00a972:	882a                	mv	a6,a0
1c00a974:	b5b9                	j	1c00a7c2 <__subdf3+0x90>
1c00a976:	02000e13          	li	t3,32
1c00a97a:	0117d6b3          	srl	a3,a5,a7
1c00a97e:	4601                	li	a2,0
1c00a980:	01c88863          	beq	a7,t3,1c00a990 <__subdf3+0x25e>
1c00a984:	04000613          	li	a2,64
1c00a988:	411608b3          	sub	a7,a2,a7
1c00a98c:	01179633          	sll	a2,a5,a7
1c00a990:	8d51                	or	a0,a0,a2
1c00a992:	00a03533          	snez	a0,a0
1c00a996:	8d55                	or	a0,a0,a3
1c00a998:	4881                	li	a7,0
1c00a99a:	b7f1                	j	1c00a966 <__subdf3+0x234>
1c00a99c:	8d5d                	or	a0,a0,a5
1c00a99e:	00a03533          	snez	a0,a0
1c00a9a2:	bfdd                	j	1c00a998 <__subdf3+0x266>
1c00a9a4:	001e8693          	addi	a3,t4,1
1c00a9a8:	e8b6be33          	p.bclr	t3,a3,20,11
1c00a9ac:	4605                	li	a2,1
1c00a9ae:	0fc64763          	blt	a2,t3,1c00aa9c <__subdf3+0x36a>
1c00a9b2:	00a7e6b3          	or	a3,a5,a0
1c00a9b6:	040e9263          	bnez	t4,1c00a9fa <__subdf3+0x2c8>
1c00a9ba:	26068a63          	beqz	a3,1c00ac2e <__subdf3+0x4fc>
1c00a9be:	010766b3          	or	a3,a4,a6
1c00a9c2:	52068a63          	beqz	a3,1c00aef6 <__subdf3+0x7c4>
1c00a9c6:	982a                	add	a6,a6,a0
1c00a9c8:	97ba                	add	a5,a5,a4
1c00a9ca:	00a83533          	sltu	a0,a6,a0
1c00a9ce:	00a78733          	add	a4,a5,a0
1c00a9d2:	00871e13          	slli	t3,a4,0x8
1c00a9d6:	240e5c63          	bgez	t3,1c00ac2e <__subdf3+0x4fc>
1c00a9da:	c17737b3          	p.bclr	a5,a4,0,23
1c00a9de:	8542                	mv	a0,a6
1c00a9e0:	4e01                	li	t3,0
1c00a9e2:	4f05                	li	t5,1
1c00a9e4:	f8353733          	p.bclr	a4,a0,28,3
1c00a9e8:	e77d                	bnez	a4,1c00aad6 <__subdf3+0x3a4>
1c00a9ea:	04088b63          	beqz	a7,1c00aa40 <__subdf3+0x30e>
1c00a9ee:	fc1e3733          	p.bclr	a4,t3,30,1
1c00a9f2:	c739                	beqz	a4,1c00aa40 <__subdf3+0x30e>
1c00a9f4:	c01e4e33          	p.bset	t3,t3,0,1
1c00a9f8:	a0a1                	j	1c00aa40 <__subdf3+0x30e>
1c00a9fa:	7ff00613          	li	a2,2047
1c00a9fe:	08ce9a63          	bne	t4,a2,1c00aa92 <__subdf3+0x360>
1c00aa02:	58068a63          	beqz	a3,1c00af96 <__subdf3+0x864>
1c00aa06:	00400e37          	lui	t3,0x400
1c00aa0a:	01c7fe33          	and	t3,a5,t3
1c00aa0e:	001e3e13          	seqz	t3,t3
1c00aa12:	0e12                	slli	t3,t3,0x4
1c00aa14:	01df1c63          	bne	t5,t4,1c00aa2c <__subdf3+0x2fa>
1c00aa18:	01076633          	or	a2,a4,a6
1c00aa1c:	c611                	beqz	a2,1c00aa28 <__subdf3+0x2f6>
1c00aa1e:	00971e93          	slli	t4,a4,0x9
1c00aa22:	000ec363          	bltz	t4,1c00aa28 <__subdf3+0x2f6>
1c00aa26:	4e41                	li	t3,16
1c00aa28:	4e068d63          	beqz	a3,1c00af22 <__subdf3+0x7f0>
1c00aa2c:	01076733          	or	a4,a4,a6
1c00aa30:	3c070663          	beqz	a4,1c00adfc <__subdf3+0x6ca>
1c00aa34:	004007b7          	lui	a5,0x400
1c00aa38:	4501                	li	a0,0
1c00aa3a:	7ff00f13          	li	t5,2047
1c00aa3e:	4581                	li	a1,0
1c00aa40:	00879613          	slli	a2,a5,0x8
1c00aa44:	00065963          	bgez	a2,1c00aa56 <__subdf3+0x324>
1c00aa48:	0f05                	addi	t5,t5,1
1c00aa4a:	7ff00713          	li	a4,2047
1c00aa4e:	50ef0f63          	beq	t5,a4,1c00af6c <__subdf3+0x83a>
1c00aa52:	c177b7b3          	p.bclr	a5,a5,0,23
1c00aa56:	01d79713          	slli	a4,a5,0x1d
1c00aa5a:	810d                	srli	a0,a0,0x3
1c00aa5c:	7ff00693          	li	a3,2047
1c00aa60:	8f49                	or	a4,a4,a0
1c00aa62:	838d                	srli	a5,a5,0x3
1c00aa64:	00df1963          	bne	t5,a3,1c00aa76 <__subdf3+0x344>
1c00aa68:	8f5d                	or	a4,a4,a5
1c00aa6a:	4781                	li	a5,0
1c00aa6c:	c709                	beqz	a4,1c00aa76 <__subdf3+0x344>
1c00aa6e:	000807b7          	lui	a5,0x80
1c00aa72:	4701                	li	a4,0
1c00aa74:	4581                	li	a1,0
1c00aa76:	4681                	li	a3,0
1c00aa78:	e607a6b3          	p.insert	a3,a5,19,0
1c00aa7c:	d54f26b3          	p.insert	a3,t5,10,20
1c00aa80:	c1f5a6b3          	p.insert	a3,a1,0,31
1c00aa84:	853a                	mv	a0,a4
1c00aa86:	85b6                	mv	a1,a3
1c00aa88:	500e0e63          	beqz	t3,1c00afa4 <__subdf3+0x872>
1c00aa8c:	001e2073          	csrs	0x1,t3
1c00aa90:	8082                	ret
1c00aa92:	4e01                	li	t3,0
1c00aa94:	f8cf1ae3          	bne	t5,a2,1c00aa28 <__subdf3+0x2f6>
1c00aa98:	4e01                	li	t3,0
1c00aa9a:	bfbd                	j	1c00aa18 <__subdf3+0x2e6>
1c00aa9c:	7ff00613          	li	a2,2047
1c00aaa0:	02c68163          	beq	a3,a2,1c00aac2 <__subdf3+0x390>
1c00aaa4:	982a                	add	a6,a6,a0
1c00aaa6:	00a83533          	sltu	a0,a6,a0
1c00aaaa:	97ba                	add	a5,a5,a4
1c00aaac:	97aa                	add	a5,a5,a0
1c00aaae:	01f79513          	slli	a0,a5,0x1f
1c00aab2:	00185813          	srli	a6,a6,0x1
1c00aab6:	01056533          	or	a0,a0,a6
1c00aaba:	8385                	srli	a5,a5,0x1
1c00aabc:	8f36                	mv	t5,a3
1c00aabe:	4e01                	li	t3,0
1c00aac0:	b715                	j	1c00a9e4 <__subdf3+0x2b2>
1c00aac2:	04030063          	beqz	t1,1c00ab02 <__subdf3+0x3d0>
1c00aac6:	02333b63          	p.bneimm	t1,3,1c00aafc <__subdf3+0x3ca>
1c00aaca:	cd85                	beqz	a1,1c00ab02 <__subdf3+0x3d0>
1c00aacc:	57fd                	li	a5,-1
1c00aace:	557d                	li	a0,-1
1c00aad0:	7fe00f13          	li	t5,2046
1c00aad4:	4e15                	li	t3,5
1c00aad6:	c00e4e33          	p.bset	t3,t3,0,0
1c00aada:	48232663          	p.beqimm	t1,2,1c00af66 <__subdf3+0x834>
1c00aade:	46332f63          	p.beqimm	t1,3,1c00af5c <__subdf3+0x82a>
1c00aae2:	f00314e3          	bnez	t1,1c00a9ea <__subdf3+0x2b8>
1c00aae6:	f6453733          	p.bclr	a4,a0,27,4
1c00aaea:	f04720e3          	p.beqimm	a4,4,1c00a9ea <__subdf3+0x2b8>
1c00aaee:	00450713          	addi	a4,a0,4
1c00aaf2:	00a73533          	sltu	a0,a4,a0
1c00aaf6:	97aa                	add	a5,a5,a0
1c00aaf8:	853a                	mv	a0,a4
1c00aafa:	bdc5                	j	1c00a9ea <__subdf3+0x2b8>
1c00aafc:	fc2338e3          	p.bneimm	t1,2,1c00aacc <__subdf3+0x39a>
1c00ab00:	d5f1                	beqz	a1,1c00aacc <__subdf3+0x39a>
1c00ab02:	7ff00f13          	li	t5,2047
1c00ab06:	a839                	j	1c00ab24 <__subdf3+0x3f2>
1c00ab08:	00030e63          	beqz	t1,1c00ab24 <__subdf3+0x3f2>
1c00ab0c:	00333963          	p.bneimm	t1,3,1c00ab1e <__subdf3+0x3ec>
1c00ab10:	c991                	beqz	a1,1c00ab24 <__subdf3+0x3f2>
1c00ab12:	57fd                	li	a5,-1
1c00ab14:	557d                	li	a0,-1
1c00ab16:	7fe00f13          	li	t5,2046
1c00ab1a:	4881                	li	a7,0
1c00ab1c:	bf65                	j	1c00aad4 <__subdf3+0x3a2>
1c00ab1e:	fe233ae3          	p.bneimm	t1,2,1c00ab12 <__subdf3+0x3e0>
1c00ab22:	d9e5                	beqz	a1,1c00ab12 <__subdf3+0x3e0>
1c00ab24:	4781                	li	a5,0
1c00ab26:	4501                	li	a0,0
1c00ab28:	4e15                	li	t3,5
1c00ab2a:	bf19                	j	1c00aa40 <__subdf3+0x30e>
1c00ab2c:	0f105363          	blez	a7,1c00ac12 <__subdf3+0x4e0>
1c00ab30:	080f1663          	bnez	t5,1c00abbc <__subdf3+0x48a>
1c00ab34:	010766b3          	or	a3,a4,a6
1c00ab38:	e699                	bnez	a3,1c00ab46 <__subdf3+0x414>
1c00ab3a:	7ff00713          	li	a4,2047
1c00ab3e:	cae88ee3          	beq	a7,a4,1c00a7fa <__subdf3+0xc8>
1c00ab42:	8f46                	mv	t5,a7
1c00ab44:	b16d                	j	1c00a7ee <__subdf3+0xbc>
1c00ab46:	fff88693          	addi	a3,a7,-1
1c00ab4a:	e295                	bnez	a3,1c00ab6e <__subdf3+0x43c>
1c00ab4c:	41050833          	sub	a6,a0,a6
1c00ab50:	8f99                	sub	a5,a5,a4
1c00ab52:	01053533          	sltu	a0,a0,a6
1c00ab56:	40a78733          	sub	a4,a5,a0
1c00ab5a:	4f05                	li	t5,1
1c00ab5c:	00871e93          	slli	t4,a4,0x8
1c00ab60:	0c0ed563          	bgez	t4,1c00ac2a <__subdf3+0x4f8>
1c00ab64:	d1773633          	p.bclr	a2,a4,8,23
1c00ab68:	88c2                	mv	a7,a6
1c00ab6a:	8efa                	mv	t4,t5
1c00ab6c:	a4f9                	j	1c00ae3a <__subdf3+0x708>
1c00ab6e:	7ff00613          	li	a2,2047
1c00ab72:	cac885e3          	beq	a7,a2,1c00a81c <__subdf3+0xea>
1c00ab76:	03800613          	li	a2,56
1c00ab7a:	08d64763          	blt	a2,a3,1c00ac08 <__subdf3+0x4d6>
1c00ab7e:	467d                	li	a2,31
1c00ab80:	04d64f63          	blt	a2,a3,1c00abde <__subdf3+0x4ac>
1c00ab84:	02000893          	li	a7,32
1c00ab88:	40d888b3          	sub	a7,a7,a3
1c00ab8c:	00d85e33          	srl	t3,a6,a3
1c00ab90:	01171633          	sll	a2,a4,a7
1c00ab94:	01181833          	sll	a6,a6,a7
1c00ab98:	01c66633          	or	a2,a2,t3
1c00ab9c:	01003833          	snez	a6,a6
1c00aba0:	01066833          	or	a6,a2,a6
1c00aba4:	00d756b3          	srl	a3,a4,a3
1c00aba8:	41050833          	sub	a6,a0,a6
1c00abac:	40d786b3          	sub	a3,a5,a3
1c00abb0:	010537b3          	sltu	a5,a0,a6
1c00abb4:	40f68733          	sub	a4,a3,a5
1c00abb8:	8f76                	mv	t5,t4
1c00abba:	b74d                	j	1c00ab5c <__subdf3+0x42a>
1c00abbc:	7ff00693          	li	a3,2047
1c00abc0:	00de9b63          	bne	t4,a3,1c00abd6 <__subdf3+0x4a4>
1c00abc4:	00a7e733          	or	a4,a5,a0
1c00abc8:	36070063          	beqz	a4,1c00af28 <__subdf3+0x7f6>
1c00abcc:	00979e93          	slli	t4,a5,0x9
1c00abd0:	7ff00f13          	li	t5,2047
1c00abd4:	b309                	j	1c00a8d6 <__subdf3+0x1a4>
1c00abd6:	c1774733          	p.bset	a4,a4,0,23
1c00abda:	86c6                	mv	a3,a7
1c00abdc:	bf69                	j	1c00ab76 <__subdf3+0x444>
1c00abde:	02000e13          	li	t3,32
1c00abe2:	00d75633          	srl	a2,a4,a3
1c00abe6:	4881                	li	a7,0
1c00abe8:	01c68863          	beq	a3,t3,1c00abf8 <__subdf3+0x4c6>
1c00abec:	04000893          	li	a7,64
1c00abf0:	40d886b3          	sub	a3,a7,a3
1c00abf4:	00d718b3          	sll	a7,a4,a3
1c00abf8:	0108e833          	or	a6,a7,a6
1c00abfc:	01003833          	snez	a6,a6
1c00ac00:	01066833          	or	a6,a2,a6
1c00ac04:	4681                	li	a3,0
1c00ac06:	b74d                	j	1c00aba8 <__subdf3+0x476>
1c00ac08:	01076833          	or	a6,a4,a6
1c00ac0c:	01003833          	snez	a6,a6
1c00ac10:	bfd5                	j	1c00ac04 <__subdf3+0x4d2>
1c00ac12:	0e088a63          	beqz	a7,1c00ad06 <__subdf3+0x5d4>
1c00ac16:	040e9d63          	bnez	t4,1c00ac70 <__subdf3+0x53e>
1c00ac1a:	00a7e633          	or	a2,a5,a0
1c00ac1e:	ea11                	bnez	a2,1c00ac32 <__subdf3+0x500>
1c00ac20:	7ff00793          	li	a5,2047
1c00ac24:	85b6                	mv	a1,a3
1c00ac26:	ccff0ce3          	beq	t5,a5,1c00a8fe <__subdf3+0x1cc>
1c00ac2a:	ce0f1fe3          	bnez	t5,1c00a928 <__subdf3+0x1f6>
1c00ac2e:	8542                	mv	a0,a6
1c00ac30:	a211                	j	1c00ad34 <__subdf3+0x602>
1c00ac32:	01f8bd63          	p.bneimm	a7,-1,1c00ac4c <__subdf3+0x51a>
1c00ac36:	40a805b3          	sub	a1,a6,a0
1c00ac3a:	40f707b3          	sub	a5,a4,a5
1c00ac3e:	00b83733          	sltu	a4,a6,a1
1c00ac42:	40e78733          	sub	a4,a5,a4
1c00ac46:	882e                	mv	a6,a1
1c00ac48:	85b6                	mv	a1,a3
1c00ac4a:	bf09                	j	1c00ab5c <__subdf3+0x42a>
1c00ac4c:	7ff00613          	li	a2,2047
1c00ac50:	fff8c893          	not	a7,a7
1c00ac54:	04cf1063          	bne	t5,a2,1c00ac94 <__subdf3+0x562>
1c00ac58:	01076533          	or	a0,a4,a6
1c00ac5c:	2e050263          	beqz	a0,1c00af40 <__subdf3+0x80e>
1c00ac60:	00971793          	slli	a5,a4,0x9
1c00ac64:	2c07da63          	bgez	a5,1c00af38 <__subdf3+0x806>
1c00ac68:	87ba                	mv	a5,a4
1c00ac6a:	8542                	mv	a0,a6
1c00ac6c:	85b6                	mv	a1,a3
1c00ac6e:	b641                	j	1c00a7ee <__subdf3+0xbc>
1c00ac70:	7ff00613          	li	a2,2047
1c00ac74:	00cf1c63          	bne	t5,a2,1c00ac8c <__subdf3+0x55a>
1c00ac78:	01076533          	or	a0,a4,a6
1c00ac7c:	2c050263          	beqz	a0,1c00af40 <__subdf3+0x80e>
1c00ac80:	00971e13          	slli	t3,a4,0x9
1c00ac84:	87ba                	mv	a5,a4
1c00ac86:	8542                	mv	a0,a6
1c00ac88:	85b6                	mv	a1,a3
1c00ac8a:	b64d                	j	1c00a82c <__subdf3+0xfa>
1c00ac8c:	411008b3          	neg	a7,a7
1c00ac90:	c177c7b3          	p.bset	a5,a5,0,23
1c00ac94:	03800613          	li	a2,56
1c00ac98:	07164363          	blt	a2,a7,1c00acfe <__subdf3+0x5cc>
1c00ac9c:	467d                	li	a2,31
1c00ac9e:	03164d63          	blt	a2,a7,1c00acd8 <__subdf3+0x5a6>
1c00aca2:	02000593          	li	a1,32
1c00aca6:	411585b3          	sub	a1,a1,a7
1c00acaa:	01155e33          	srl	t3,a0,a7
1c00acae:	00b79633          	sll	a2,a5,a1
1c00acb2:	00b51533          	sll	a0,a0,a1
1c00acb6:	01c66633          	or	a2,a2,t3
1c00acba:	00a03533          	snez	a0,a0
1c00acbe:	8d51                	or	a0,a0,a2
1c00acc0:	0117d8b3          	srl	a7,a5,a7
1c00acc4:	40a80533          	sub	a0,a6,a0
1c00acc8:	411707b3          	sub	a5,a4,a7
1c00accc:	00a83733          	sltu	a4,a6,a0
1c00acd0:	40e78733          	sub	a4,a5,a4
1c00acd4:	882a                	mv	a6,a0
1c00acd6:	bf8d                	j	1c00ac48 <__subdf3+0x516>
1c00acd8:	02000e13          	li	t3,32
1c00acdc:	0117d633          	srl	a2,a5,a7
1c00ace0:	4581                	li	a1,0
1c00ace2:	01c88863          	beq	a7,t3,1c00acf2 <__subdf3+0x5c0>
1c00ace6:	04000593          	li	a1,64
1c00acea:	411588b3          	sub	a7,a1,a7
1c00acee:	011795b3          	sll	a1,a5,a7
1c00acf2:	8d4d                	or	a0,a0,a1
1c00acf4:	00a03533          	snez	a0,a0
1c00acf8:	8d51                	or	a0,a0,a2
1c00acfa:	4881                	li	a7,0
1c00acfc:	b7e1                	j	1c00acc4 <__subdf3+0x592>
1c00acfe:	8d5d                	or	a0,a0,a5
1c00ad00:	00a03533          	snez	a0,a0
1c00ad04:	bfdd                	j	1c00acfa <__subdf3+0x5c8>
1c00ad06:	001e8613          	addi	a2,t4,1
1c00ad0a:	e8b63633          	p.bclr	a2,a2,20,11
1c00ad0e:	4e05                	li	t3,1
1c00ad10:	10ce4063          	blt	t3,a2,1c00ae10 <__subdf3+0x6de>
1c00ad14:	00a7efb3          	or	t6,a5,a0
1c00ad18:	01076633          	or	a2,a4,a6
1c00ad1c:	0a0e9463          	bnez	t4,1c00adc4 <__subdf3+0x692>
1c00ad20:	060f9063          	bnez	t6,1c00ad80 <__subdf3+0x64e>
1c00ad24:	1c061763          	bnez	a2,1c00aef2 <__subdf3+0x7c0>
1c00ad28:	ffe30593          	addi	a1,t1,-2
1c00ad2c:	0015b593          	seqz	a1,a1
1c00ad30:	4701                	li	a4,0
1c00ad32:	4501                	li	a0,0
1c00ad34:	00e567b3          	or	a5,a0,a4
1c00ad38:	20078f63          	beqz	a5,1c00af56 <__subdf3+0x824>
1c00ad3c:	01f55793          	srli	a5,a0,0x1f
1c00ad40:	00171893          	slli	a7,a4,0x1
1c00ad44:	98be                	add	a7,a7,a5
1c00ad46:	00151793          	slli	a5,a0,0x1
1c00ad4a:	f837b6b3          	p.bclr	a3,a5,28,3
1c00ad4e:	4e01                	li	t3,0
1c00ad50:	ce99                	beqz	a3,1c00ad6e <__subdf3+0x63c>
1c00ad52:	1c232063          	p.beqimm	t1,2,1c00af12 <__subdf3+0x7e0>
1c00ad56:	1a332563          	p.beqimm	t1,3,1c00af00 <__subdf3+0x7ce>
1c00ad5a:	4e05                	li	t3,1
1c00ad5c:	00031963          	bnez	t1,1c00ad6e <__subdf3+0x63c>
1c00ad60:	f647b6b3          	p.bclr	a3,a5,27,4
1c00ad64:	4e05                	li	t3,1
1c00ad66:	ffc7b793          	sltiu	a5,a5,-4
1c00ad6a:	1a46b063          	p.bneimm	a3,4,1c00af0a <__subdf3+0x7d8>
1c00ad6e:	0188d893          	srli	a7,a7,0x18
1c00ad72:	0018c893          	xori	a7,a7,1
1c00ad76:	fc18b8b3          	p.bclr	a7,a7,30,1
1c00ad7a:	87ba                	mv	a5,a4
1c00ad7c:	4f01                	li	t5,0
1c00ad7e:	b19d                	j	1c00a9e4 <__subdf3+0x2b2>
1c00ad80:	16060b63          	beqz	a2,1c00aef6 <__subdf3+0x7c4>
1c00ad84:	410508b3          	sub	a7,a0,a6
1c00ad88:	40e78633          	sub	a2,a5,a4
1c00ad8c:	01153e33          	sltu	t3,a0,a7
1c00ad90:	41c60633          	sub	a2,a2,t3
1c00ad94:	00861e93          	slli	t4,a2,0x8
1c00ad98:	000edc63          	bgez	t4,1c00adb0 <__subdf3+0x67e>
1c00ad9c:	40a80533          	sub	a0,a6,a0
1c00ada0:	40f707b3          	sub	a5,a4,a5
1c00ada4:	00a83733          	sltu	a4,a6,a0
1c00ada8:	40e78733          	sub	a4,a5,a4
1c00adac:	85b6                	mv	a1,a3
1c00adae:	b759                	j	1c00ad34 <__subdf3+0x602>
1c00adb0:	00c8e533          	or	a0,a7,a2
1c00adb4:	14051363          	bnez	a0,1c00aefa <__subdf3+0x7c8>
1c00adb8:	ffe30593          	addi	a1,t1,-2
1c00adbc:	0015b593          	seqz	a1,a1
1c00adc0:	4701                	li	a4,0
1c00adc2:	bf8d                	j	1c00ad34 <__subdf3+0x602>
1c00adc4:	7ff00293          	li	t0,2047
1c00adc8:	025e9d63          	bne	t4,t0,1c00ae02 <__subdf3+0x6d0>
1c00adcc:	1c0f8163          	beqz	t6,1c00af8e <__subdf3+0x85c>
1c00add0:	00400e37          	lui	t3,0x400
1c00add4:	01c7fe33          	and	t3,a5,t3
1c00add8:	001e3e13          	seqz	t3,t3
1c00addc:	0e12                	slli	t3,t3,0x4
1c00adde:	03df1763          	bne	t5,t4,1c00ae0c <__subdf3+0x6da>
1c00ade2:	c611                	beqz	a2,1c00adee <__subdf3+0x6bc>
1c00ade4:	00971e93          	slli	t4,a4,0x9
1c00ade8:	000ec363          	bltz	t4,1c00adee <__subdf3+0x6bc>
1c00adec:	4e41                	li	t3,16
1c00adee:	000f9f63          	bnez	t6,1c00ae0c <__subdf3+0x6da>
1c00adf2:	14060a63          	beqz	a2,1c00af46 <__subdf3+0x814>
1c00adf6:	87ba                	mv	a5,a4
1c00adf8:	8542                	mv	a0,a6
1c00adfa:	85b6                	mv	a1,a3
1c00adfc:	7ff00f13          	li	t5,2047
1c00ae00:	b6d5                	j	1c00a9e4 <__subdf3+0x2b2>
1c00ae02:	4e01                	li	t3,0
1c00ae04:	fe5f15e3          	bne	t5,t0,1c00adee <__subdf3+0x6bc>
1c00ae08:	4e01                	li	t3,0
1c00ae0a:	bfe1                	j	1c00ade2 <__subdf3+0x6b0>
1c00ae0c:	da65                	beqz	a2,1c00adfc <__subdf3+0x6ca>
1c00ae0e:	b11d                	j	1c00aa34 <__subdf3+0x302>
1c00ae10:	410508b3          	sub	a7,a0,a6
1c00ae14:	01153e33          	sltu	t3,a0,a7
1c00ae18:	40e78633          	sub	a2,a5,a4
1c00ae1c:	41c60633          	sub	a2,a2,t3
1c00ae20:	00861e13          	slli	t3,a2,0x8
1c00ae24:	060e5c63          	bgez	t3,1c00ae9c <__subdf3+0x76a>
1c00ae28:	40a808b3          	sub	a7,a6,a0
1c00ae2c:	40f707b3          	sub	a5,a4,a5
1c00ae30:	01183733          	sltu	a4,a6,a7
1c00ae34:	40e78633          	sub	a2,a5,a4
1c00ae38:	85b6                	mv	a1,a3
1c00ae3a:	c62d                	beqz	a2,1c00aea4 <__subdf3+0x772>
1c00ae3c:	100616b3          	p.fl1	a3,a2
1c00ae40:	47fd                	li	a5,31
1c00ae42:	40d786b3          	sub	a3,a5,a3
1c00ae46:	ff868f13          	addi	t5,a3,-8
1c00ae4a:	47fd                	li	a5,31
1c00ae4c:	07e7c463          	blt	a5,t5,1c00aeb4 <__subdf3+0x782>
1c00ae50:	02000793          	li	a5,32
1c00ae54:	41e787b3          	sub	a5,a5,t5
1c00ae58:	01e61633          	sll	a2,a2,t5
1c00ae5c:	00f8d733          	srl	a4,a7,a5
1c00ae60:	00c767b3          	or	a5,a4,a2
1c00ae64:	01e89833          	sll	a6,a7,t5
1c00ae68:	09df4063          	blt	t5,t4,1c00aee8 <__subdf3+0x7b6>
1c00ae6c:	41df0f33          	sub	t5,t5,t4
1c00ae70:	001f0713          	addi	a4,t5,1
1c00ae74:	46fd                	li	a3,31
1c00ae76:	04e6c563          	blt	a3,a4,1c00aec0 <__subdf3+0x78e>
1c00ae7a:	02000693          	li	a3,32
1c00ae7e:	8e99                	sub	a3,a3,a4
1c00ae80:	00e85633          	srl	a2,a6,a4
1c00ae84:	00d79533          	sll	a0,a5,a3
1c00ae88:	00d81833          	sll	a6,a6,a3
1c00ae8c:	8d51                	or	a0,a0,a2
1c00ae8e:	01003833          	snez	a6,a6
1c00ae92:	01056533          	or	a0,a0,a6
1c00ae96:	00e7d733          	srl	a4,a5,a4
1c00ae9a:	bd69                	j	1c00ad34 <__subdf3+0x602>
1c00ae9c:	00c8e533          	or	a0,a7,a2
1c00aea0:	fd49                	bnez	a0,1c00ae3a <__subdf3+0x708>
1c00aea2:	bf19                	j	1c00adb8 <__subdf3+0x686>
1c00aea4:	100896b3          	p.fl1	a3,a7
1c00aea8:	47fd                	li	a5,31
1c00aeaa:	40d786b3          	sub	a3,a5,a3
1c00aeae:	02068693          	addi	a3,a3,32
1c00aeb2:	bf51                	j	1c00ae46 <__subdf3+0x714>
1c00aeb4:	fd868793          	addi	a5,a3,-40
1c00aeb8:	00f897b3          	sll	a5,a7,a5
1c00aebc:	4801                	li	a6,0
1c00aebe:	b76d                	j	1c00ae68 <__subdf3+0x736>
1c00aec0:	1f05                	addi	t5,t5,-31
1c00aec2:	02000693          	li	a3,32
1c00aec6:	01e7df33          	srl	t5,a5,t5
1c00aeca:	4601                	li	a2,0
1c00aecc:	00d70763          	beq	a4,a3,1c00aeda <__subdf3+0x7a8>
1c00aed0:	04000613          	li	a2,64
1c00aed4:	8e19                	sub	a2,a2,a4
1c00aed6:	00c79633          	sll	a2,a5,a2
1c00aeda:	00c86533          	or	a0,a6,a2
1c00aede:	00a03533          	snez	a0,a0
1c00aee2:	00af6533          	or	a0,t5,a0
1c00aee6:	bde9                	j	1c00adc0 <__subdf3+0x68e>
1c00aee8:	41ee8f33          	sub	t5,t4,t5
1c00aeec:	c177b733          	p.bclr	a4,a5,0,23
1c00aef0:	bb2d                	j	1c00ac2a <__subdf3+0x4f8>
1c00aef2:	8542                	mv	a0,a6
1c00aef4:	bd65                	j	1c00adac <__subdf3+0x67a>
1c00aef6:	873e                	mv	a4,a5
1c00aef8:	bd35                	j	1c00ad34 <__subdf3+0x602>
1c00aefa:	8732                	mv	a4,a2
1c00aefc:	8546                	mv	a0,a7
1c00aefe:	bd1d                	j	1c00ad34 <__subdf3+0x602>
1c00af00:	4e05                	li	t3,1
1c00af02:	e60596e3          	bnez	a1,1c00ad6e <__subdf3+0x63c>
1c00af06:	ff87b793          	sltiu	a5,a5,-8
1c00af0a:	0017c793          	xori	a5,a5,1
1c00af0e:	98be                	add	a7,a7,a5
1c00af10:	bdb9                	j	1c00ad6e <__subdf3+0x63c>
1c00af12:	4e05                	li	t3,1
1c00af14:	e4058de3          	beqz	a1,1c00ad6e <__subdf3+0x63c>
1c00af18:	b7fd                	j	1c00af06 <__subdf3+0x7d4>
1c00af1a:	87ba                	mv	a5,a4
1c00af1c:	8542                	mv	a0,a6
1c00af1e:	4881                	li	a7,0
1c00af20:	b8c5                	j	1c00a810 <__subdf3+0xde>
1c00af22:	87ba                	mv	a5,a4
1c00af24:	8542                	mv	a0,a6
1c00af26:	bdd9                	j	1c00adfc <__subdf3+0x6ca>
1c00af28:	4781                	li	a5,0
1c00af2a:	4501                	li	a0,0
1c00af2c:	7ff00f13          	li	t5,2047
1c00af30:	4e01                	li	t3,0
1c00af32:	b639                	j	1c00aa40 <__subdf3+0x30e>
1c00af34:	4781                	li	a5,0
1c00af36:	bfed                	j	1c00af30 <__subdf3+0x7fe>
1c00af38:	87ba                	mv	a5,a4
1c00af3a:	8542                	mv	a0,a6
1c00af3c:	85b6                	mv	a1,a3
1c00af3e:	b7c5                	j	1c00af1e <__subdf3+0x7ec>
1c00af40:	4781                	li	a5,0
1c00af42:	85b6                	mv	a1,a3
1c00af44:	b7f5                	j	1c00af30 <__subdf3+0x7fe>
1c00af46:	4501                	li	a0,0
1c00af48:	004007b7          	lui	a5,0x400
1c00af4c:	7ff00f13          	li	t5,2047
1c00af50:	4581                	li	a1,0
1c00af52:	4e41                	li	t3,16
1c00af54:	b4f5                	j	1c00aa40 <__subdf3+0x30e>
1c00af56:	4501                	li	a0,0
1c00af58:	4f01                	li	t5,0
1c00af5a:	bfd9                	j	1c00af30 <__subdf3+0x7fe>
1c00af5c:	a80597e3          	bnez	a1,1c00a9ea <__subdf3+0x2b8>
1c00af60:	00850713          	addi	a4,a0,8
1c00af64:	b679                	j	1c00aaf2 <__subdf3+0x3c0>
1c00af66:	a80582e3          	beqz	a1,1c00a9ea <__subdf3+0x2b8>
1c00af6a:	bfdd                	j	1c00af60 <__subdf3+0x82e>
1c00af6c:	4501                	li	a0,0
1c00af6e:	00030c63          	beqz	t1,1c00af86 <__subdf3+0x854>
1c00af72:	00333763          	p.bneimm	t1,3,1c00af80 <__subdf3+0x84e>
1c00af76:	c981                	beqz	a1,1c00af86 <__subdf3+0x854>
1c00af78:	557d                	li	a0,-1
1c00af7a:	7fe00f13          	li	t5,2046
1c00af7e:	a021                	j	1c00af86 <__subdf3+0x854>
1c00af80:	fe233ce3          	p.bneimm	t1,2,1c00af78 <__subdf3+0x846>
1c00af84:	d9f5                	beqz	a1,1c00af78 <__subdf3+0x846>
1c00af86:	005e6e13          	ori	t3,t3,5
1c00af8a:	87aa                	mv	a5,a0
1c00af8c:	b4e9                	j	1c00aa56 <__subdf3+0x324>
1c00af8e:	4e01                	li	t3,0
1c00af90:	e7df11e3          	bne	t5,t4,1c00adf2 <__subdf3+0x6c0>
1c00af94:	bd95                	j	1c00ae08 <__subdf3+0x6d6>
1c00af96:	b1df01e3          	beq	t5,t4,1c00aa98 <__subdf3+0x366>
1c00af9a:	87ba                	mv	a5,a4
1c00af9c:	8542                	mv	a0,a6
1c00af9e:	7ff00f13          	li	t5,2047
1c00afa2:	be31                	j	1c00aabe <__subdf3+0x38c>
1c00afa4:	8082                	ret

1c00afa6 <__fixdfsi>:
1c00afa6:	002027f3          	csrr	a5,0x2
1c00afaa:	3fe00793          	li	a5,1022
1c00afae:	d54596b3          	p.extractu	a3,a1,10,20
1c00afb2:	e6059633          	p.extractu	a2,a1,19,0
1c00afb6:	81fd                	srli	a1,a1,0x1f
1c00afb8:	00d7c963          	blt	a5,a3,1c00afca <__fixdfsi+0x24>
1c00afbc:	eec1                	bnez	a3,1c00b054 <__fixdfsi+0xae>
1c00afbe:	00a66733          	or	a4,a2,a0
1c00afc2:	4781                	li	a5,0
1c00afc4:	eb49                	bnez	a4,1c00b056 <__fixdfsi+0xb0>
1c00afc6:	853e                	mv	a0,a5
1c00afc8:	8082                	ret
1c00afca:	41d00793          	li	a5,1053
1c00afce:	02d7d663          	ble	a3,a5,1c00affa <__fixdfsi+0x54>
1c00afd2:	80000737          	lui	a4,0x80000
1c00afd6:	fff74713          	not	a4,a4
1c00afda:	00e587b3          	add	a5,a1,a4
1c00afde:	cdb5                	beqz	a1,1c00b05a <__fixdfsi+0xb4>
1c00afe0:	41e00593          	li	a1,1054
1c00afe4:	4741                	li	a4,16
1c00afe6:	04b69163          	bne	a3,a1,1c00b028 <__fixdfsi+0x82>
1c00afea:	062e                	slli	a2,a2,0xb
1c00afec:	01555693          	srli	a3,a0,0x15
1c00aff0:	8e55                	or	a2,a2,a3
1c00aff2:	ea1d                	bnez	a2,1c00b028 <__fixdfsi+0x82>
1c00aff4:	00b51713          	slli	a4,a0,0xb
1c00aff8:	b7f1                	j	1c00afc4 <__fixdfsi+0x1e>
1c00affa:	c14647b3          	p.bset	a5,a2,0,20
1c00affe:	43300613          	li	a2,1075
1c00b002:	8e15                	sub	a2,a2,a3
1c00b004:	477d                	li	a4,31
1c00b006:	02c74463          	blt	a4,a2,1c00b02e <__fixdfsi+0x88>
1c00b00a:	bed68693          	addi	a3,a3,-1043
1c00b00e:	00d51733          	sll	a4,a0,a3
1c00b012:	00d797b3          	sll	a5,a5,a3
1c00b016:	00c55533          	srl	a0,a0,a2
1c00b01a:	00e03733          	snez	a4,a4
1c00b01e:	8fc9                	or	a5,a5,a0
1c00b020:	c199                	beqz	a1,1c00b026 <__fixdfsi+0x80>
1c00b022:	40f007b3          	neg	a5,a5
1c00b026:	d345                	beqz	a4,1c00afc6 <__fixdfsi+0x20>
1c00b028:	00172073          	csrs	0x1,a4
1c00b02c:	bf69                	j	1c00afc6 <__fixdfsi+0x20>
1c00b02e:	02000813          	li	a6,32
1c00b032:	4701                	li	a4,0
1c00b034:	01060663          	beq	a2,a6,1c00b040 <__fixdfsi+0x9a>
1c00b038:	c0d68713          	addi	a4,a3,-1011
1c00b03c:	00e79733          	sll	a4,a5,a4
1c00b040:	8f49                	or	a4,a4,a0
1c00b042:	41300513          	li	a0,1043
1c00b046:	40d506b3          	sub	a3,a0,a3
1c00b04a:	00e03733          	snez	a4,a4
1c00b04e:	00d7d7b3          	srl	a5,a5,a3
1c00b052:	b7f9                	j	1c00b020 <__fixdfsi+0x7a>
1c00b054:	4781                	li	a5,0
1c00b056:	4705                	li	a4,1
1c00b058:	bfc1                	j	1c00b028 <__fixdfsi+0x82>
1c00b05a:	4741                	li	a4,16
1c00b05c:	b7f1                	j	1c00b028 <__fixdfsi+0x82>

1c00b05e <__floatsidf>:
1c00b05e:	c539                	beqz	a0,1c00b0ac <__floatsidf+0x4e>
1c00b060:	01f55613          	srli	a2,a0,0x1f
1c00b064:	00055463          	bgez	a0,1c00b06c <__floatsidf+0xe>
1c00b068:	40a00533          	neg	a0,a0
1c00b06c:	477d                	li	a4,31
1c00b06e:	100517b3          	p.fl1	a5,a0
1c00b072:	40f707b3          	sub	a5,a4,a5
1c00b076:	41e00693          	li	a3,1054
1c00b07a:	4729                	li	a4,10
1c00b07c:	8e9d                	sub	a3,a3,a5
1c00b07e:	02f74263          	blt	a4,a5,1c00b0a2 <__floatsidf+0x44>
1c00b082:	472d                	li	a4,11
1c00b084:	8f1d                	sub	a4,a4,a5
1c00b086:	07d5                	addi	a5,a5,21
1c00b088:	00e55733          	srl	a4,a0,a4
1c00b08c:	00f51533          	sll	a0,a0,a5
1c00b090:	4781                	li	a5,0
1c00b092:	e60727b3          	p.insert	a5,a4,19,0
1c00b096:	d546a7b3          	p.insert	a5,a3,10,20
1c00b09a:	c1f627b3          	p.insert	a5,a2,0,31
1c00b09e:	85be                	mv	a1,a5
1c00b0a0:	8082                	ret
1c00b0a2:	17d5                	addi	a5,a5,-11
1c00b0a4:	00f51733          	sll	a4,a0,a5
1c00b0a8:	4501                	li	a0,0
1c00b0aa:	b7dd                	j	1c00b090 <__floatsidf+0x32>
1c00b0ac:	4701                	li	a4,0
1c00b0ae:	4501                	li	a0,0
1c00b0b0:	4681                	li	a3,0
1c00b0b2:	4601                	li	a2,0
1c00b0b4:	bff1                	j	1c00b090 <__floatsidf+0x32>

1c00b0b6 <__extendsfdf2>:
1c00b0b6:	002027f3          	csrr	a5,0x2
1c00b0ba:	cf751633          	p.extractu	a2,a0,7,23
1c00b0be:	00160713          	addi	a4,a2,1 # 10001 <_l1_preload_size+0xf681>
1c00b0c2:	ee873733          	p.bclr	a4,a4,23,8
1c00b0c6:	4685                	li	a3,1
1c00b0c8:	ec0517b3          	p.extractu	a5,a0,22,0
1c00b0cc:	817d                	srli	a0,a0,0x1f
1c00b0ce:	00e6d963          	ble	a4,a3,1c00b0e0 <__extendsfdf2+0x2a>
1c00b0d2:	0037d713          	srli	a4,a5,0x3
1c00b0d6:	38060613          	addi	a2,a2,896
1c00b0da:	07f6                	slli	a5,a5,0x1d
1c00b0dc:	4681                	li	a3,0
1c00b0de:	a899                	j	1c00b134 <__extendsfdf2+0x7e>
1c00b0e0:	ee05                	bnez	a2,1c00b118 <__extendsfdf2+0x62>
1c00b0e2:	c7ad                	beqz	a5,1c00b14c <__extendsfdf2+0x96>
1c00b0e4:	477d                	li	a4,31
1c00b0e6:	100796b3          	p.fl1	a3,a5
1c00b0ea:	40d706b3          	sub	a3,a4,a3
1c00b0ee:	4729                	li	a4,10
1c00b0f0:	00d74e63          	blt	a4,a3,1c00b10c <__extendsfdf2+0x56>
1c00b0f4:	472d                	li	a4,11
1c00b0f6:	8f15                	sub	a4,a4,a3
1c00b0f8:	01568613          	addi	a2,a3,21
1c00b0fc:	00e7d733          	srl	a4,a5,a4
1c00b100:	00c797b3          	sll	a5,a5,a2
1c00b104:	38900613          	li	a2,905
1c00b108:	8e15                	sub	a2,a2,a3
1c00b10a:	bfc9                	j	1c00b0dc <__extendsfdf2+0x26>
1c00b10c:	ff568713          	addi	a4,a3,-11
1c00b110:	00e79733          	sll	a4,a5,a4
1c00b114:	4781                	li	a5,0
1c00b116:	b7fd                	j	1c00b104 <__extendsfdf2+0x4e>
1c00b118:	cf8d                	beqz	a5,1c00b152 <__extendsfdf2+0x9c>
1c00b11a:	004006b7          	lui	a3,0x400
1c00b11e:	8efd                	and	a3,a3,a5
1c00b120:	0037d713          	srli	a4,a5,0x3
1c00b124:	0016b693          	seqz	a3,a3
1c00b128:	0692                	slli	a3,a3,0x4
1c00b12a:	c1374733          	p.bset	a4,a4,0,19
1c00b12e:	07f6                	slli	a5,a5,0x1d
1c00b130:	7ff00613          	li	a2,2047
1c00b134:	4581                	li	a1,0
1c00b136:	e60725b3          	p.insert	a1,a4,19,0
1c00b13a:	d54625b3          	p.insert	a1,a2,10,20
1c00b13e:	c1f525b3          	p.insert	a1,a0,0,31
1c00b142:	853e                	mv	a0,a5
1c00b144:	c299                	beqz	a3,1c00b14a <__extendsfdf2+0x94>
1c00b146:	0016a073          	csrs	0x1,a3
1c00b14a:	8082                	ret
1c00b14c:	4701                	li	a4,0
1c00b14e:	4601                	li	a2,0
1c00b150:	b771                	j	1c00b0dc <__extendsfdf2+0x26>
1c00b152:	4701                	li	a4,0
1c00b154:	7ff00613          	li	a2,2047
1c00b158:	b751                	j	1c00b0dc <__extendsfdf2+0x26>

1c00b15a <__truncdfsf2>:
1c00b15a:	00202873          	csrr	a6,0x2
1c00b15e:	e6059633          	p.extractu	a2,a1,19,0
1c00b162:	d54598b3          	p.extractu	a7,a1,10,20
1c00b166:	01d55793          	srli	a5,a0,0x1d
1c00b16a:	060e                	slli	a2,a2,0x3
1c00b16c:	8e5d                	or	a2,a2,a5
1c00b16e:	00188793          	addi	a5,a7,1
1c00b172:	e8b7b7b3          	p.bclr	a5,a5,20,11
1c00b176:	4685                	li	a3,1
1c00b178:	81fd                	srli	a1,a1,0x1f
1c00b17a:	00351713          	slli	a4,a0,0x3
1c00b17e:	14f6d663          	ble	a5,a3,1c00b2ca <__truncdfsf2+0x170>
1c00b182:	c8088693          	addi	a3,a7,-896
1c00b186:	0fe00793          	li	a5,254
1c00b18a:	08d7d063          	ble	a3,a5,1c00b20a <__truncdfsf2+0xb0>
1c00b18e:	02080b63          	beqz	a6,1c00b1c4 <__truncdfsf2+0x6a>
1c00b192:	02383663          	p.bneimm	a6,3,1c00b1be <__truncdfsf2+0x64>
1c00b196:	c59d                	beqz	a1,1c00b1c4 <__truncdfsf2+0x6a>
1c00b198:	57fd                	li	a5,-1
1c00b19a:	0fe00693          	li	a3,254
1c00b19e:	4601                	li	a2,0
1c00b1a0:	4715                	li	a4,5
1c00b1a2:	c0074733          	p.bset	a4,a4,0,0
1c00b1a6:	16282d63          	p.beqimm	a6,2,1c00b320 <__truncdfsf2+0x1c6>
1c00b1aa:	16382863          	p.beqimm	a6,3,1c00b31a <__truncdfsf2+0x1c0>
1c00b1ae:	10081763          	bnez	a6,1c00b2bc <__truncdfsf2+0x162>
1c00b1b2:	f647b533          	p.bclr	a0,a5,27,4
1c00b1b6:	10452363          	p.beqimm	a0,4,1c00b2bc <__truncdfsf2+0x162>
1c00b1ba:	0791                	addi	a5,a5,4
1c00b1bc:	a201                	j	1c00b2bc <__truncdfsf2+0x162>
1c00b1be:	fc283de3          	p.bneimm	a6,2,1c00b198 <__truncdfsf2+0x3e>
1c00b1c2:	d9f9                	beqz	a1,1c00b198 <__truncdfsf2+0x3e>
1c00b1c4:	4781                	li	a5,0
1c00b1c6:	0ff00693          	li	a3,255
1c00b1ca:	4715                	li	a4,5
1c00b1cc:	00579613          	slli	a2,a5,0x5
1c00b1d0:	00065963          	bgez	a2,1c00b1e2 <__truncdfsf2+0x88>
1c00b1d4:	0685                	addi	a3,a3,1
1c00b1d6:	0ff00613          	li	a2,255
1c00b1da:	14c68563          	beq	a3,a2,1c00b324 <__truncdfsf2+0x1ca>
1c00b1de:	c1a7b7b3          	p.bclr	a5,a5,0,26
1c00b1e2:	0ff00613          	li	a2,255
1c00b1e6:	838d                	srli	a5,a5,0x3
1c00b1e8:	00c69663          	bne	a3,a2,1c00b1f4 <__truncdfsf2+0x9a>
1c00b1ec:	c781                	beqz	a5,1c00b1f4 <__truncdfsf2+0x9a>
1c00b1ee:	004007b7          	lui	a5,0x400
1c00b1f2:	4581                	li	a1,0
1c00b1f4:	4501                	li	a0,0
1c00b1f6:	ec07a533          	p.insert	a0,a5,22,0
1c00b1fa:	cf76a533          	p.insert	a0,a3,7,23
1c00b1fe:	c1f5a533          	p.insert	a0,a1,0,31
1c00b202:	c319                	beqz	a4,1c00b208 <__truncdfsf2+0xae>
1c00b204:	00172073          	csrs	0x1,a4
1c00b208:	8082                	ret
1c00b20a:	08d04b63          	bgtz	a3,1c00b2a0 <__truncdfsf2+0x146>
1c00b20e:	57a5                	li	a5,-23
1c00b210:	0ef6c863          	blt	a3,a5,1c00b300 <__truncdfsf2+0x1a6>
1c00b214:	4379                	li	t1,30
1c00b216:	40d30333          	sub	t1,t1,a3
1c00b21a:	47fd                	li	a5,31
1c00b21c:	c1764633          	p.bset	a2,a2,0,23
1c00b220:	0467cc63          	blt	a5,t1,1c00b278 <__truncdfsf2+0x11e>
1c00b224:	c8288893          	addi	a7,a7,-894
1c00b228:	011717b3          	sll	a5,a4,a7
1c00b22c:	00f037b3          	snez	a5,a5
1c00b230:	01161633          	sll	a2,a2,a7
1c00b234:	00675333          	srl	t1,a4,t1
1c00b238:	8fd1                	or	a5,a5,a2
1c00b23a:	00f367b3          	or	a5,t1,a5
1c00b23e:	4681                	li	a3,0
1c00b240:	cbf9                	beqz	a5,1c00b316 <__truncdfsf2+0x1bc>
1c00b242:	00179693          	slli	a3,a5,0x1
1c00b246:	f836b633          	p.bclr	a2,a3,28,3
1c00b24a:	4701                	li	a4,0
1c00b24c:	ce11                	beqz	a2,1c00b268 <__truncdfsf2+0x10e>
1c00b24e:	0a282f63          	p.beqimm	a6,2,1c00b30c <__truncdfsf2+0x1b2>
1c00b252:	0a382963          	p.beqimm	a6,3,1c00b304 <__truncdfsf2+0x1aa>
1c00b256:	4705                	li	a4,1
1c00b258:	00081863          	bnez	a6,1c00b268 <__truncdfsf2+0x10e>
1c00b25c:	f646b633          	p.bclr	a2,a3,27,4
1c00b260:	4705                	li	a4,1
1c00b262:	00462363          	p.beqimm	a2,4,1c00b268 <__truncdfsf2+0x10e>
1c00b266:	0691                	addi	a3,a3,4
1c00b268:	01b6d613          	srli	a2,a3,0x1b
1c00b26c:	00164613          	xori	a2,a2,1
1c00b270:	fc163633          	p.bclr	a2,a2,30,1
1c00b274:	4681                	li	a3,0
1c00b276:	a83d                	j	1c00b2b4 <__truncdfsf2+0x15a>
1c00b278:	57f9                	li	a5,-2
1c00b27a:	40d786b3          	sub	a3,a5,a3
1c00b27e:	02000793          	li	a5,32
1c00b282:	00d656b3          	srl	a3,a2,a3
1c00b286:	4501                	li	a0,0
1c00b288:	00f30663          	beq	t1,a5,1c00b294 <__truncdfsf2+0x13a>
1c00b28c:	ca288893          	addi	a7,a7,-862
1c00b290:	01161533          	sll	a0,a2,a7
1c00b294:	00e567b3          	or	a5,a0,a4
1c00b298:	00f037b3          	snez	a5,a5
1c00b29c:	8fd5                	or	a5,a5,a3
1c00b29e:	b745                	j	1c00b23e <__truncdfsf2+0xe4>
1c00b2a0:	051a                	slli	a0,a0,0x6
1c00b2a2:	00a03533          	snez	a0,a0
1c00b2a6:	060e                	slli	a2,a2,0x3
1c00b2a8:	01d75793          	srli	a5,a4,0x1d
1c00b2ac:	8e49                	or	a2,a2,a0
1c00b2ae:	8fd1                	or	a5,a5,a2
1c00b2b0:	4701                	li	a4,0
1c00b2b2:	4601                	li	a2,0
1c00b2b4:	f837b533          	p.bclr	a0,a5,28,3
1c00b2b8:	ee0515e3          	bnez	a0,1c00b1a2 <__truncdfsf2+0x48>
1c00b2bc:	da01                	beqz	a2,1c00b1cc <__truncdfsf2+0x72>
1c00b2be:	fc173633          	p.bclr	a2,a4,30,1
1c00b2c2:	d609                	beqz	a2,1c00b1cc <__truncdfsf2+0x72>
1c00b2c4:	c0174733          	p.bset	a4,a4,0,1
1c00b2c8:	b711                	j	1c00b1cc <__truncdfsf2+0x72>
1c00b2ca:	00e667b3          	or	a5,a2,a4
1c00b2ce:	00089563          	bnez	a7,1c00b2d8 <__truncdfsf2+0x17e>
1c00b2d2:	00f037b3          	snez	a5,a5
1c00b2d6:	b7a5                	j	1c00b23e <__truncdfsf2+0xe4>
1c00b2d8:	cf8d                	beqz	a5,1c00b312 <__truncdfsf2+0x1b8>
1c00b2da:	7ff00793          	li	a5,2047
1c00b2de:	4701                	li	a4,0
1c00b2e0:	00f89863          	bne	a7,a5,1c00b2f0 <__truncdfsf2+0x196>
1c00b2e4:	00400737          	lui	a4,0x400
1c00b2e8:	8f71                	and	a4,a4,a2
1c00b2ea:	00173713          	seqz	a4,a4
1c00b2ee:	0712                	slli	a4,a4,0x4
1c00b2f0:	00361793          	slli	a5,a2,0x3
1c00b2f4:	c197c7b3          	p.bset	a5,a5,0,25
1c00b2f8:	0ff00693          	li	a3,255
1c00b2fc:	4601                	li	a2,0
1c00b2fe:	bf5d                	j	1c00b2b4 <__truncdfsf2+0x15a>
1c00b300:	4785                	li	a5,1
1c00b302:	b781                	j	1c00b242 <__truncdfsf2+0xe8>
1c00b304:	4705                	li	a4,1
1c00b306:	f1ad                	bnez	a1,1c00b268 <__truncdfsf2+0x10e>
1c00b308:	06a1                	addi	a3,a3,8
1c00b30a:	bfb9                	j	1c00b268 <__truncdfsf2+0x10e>
1c00b30c:	4705                	li	a4,1
1c00b30e:	dda9                	beqz	a1,1c00b268 <__truncdfsf2+0x10e>
1c00b310:	bfe5                	j	1c00b308 <__truncdfsf2+0x1ae>
1c00b312:	0ff00693          	li	a3,255
1c00b316:	4701                	li	a4,0
1c00b318:	bd55                	j	1c00b1cc <__truncdfsf2+0x72>
1c00b31a:	f1cd                	bnez	a1,1c00b2bc <__truncdfsf2+0x162>
1c00b31c:	07a1                	addi	a5,a5,8
1c00b31e:	bf79                	j	1c00b2bc <__truncdfsf2+0x162>
1c00b320:	ddd1                	beqz	a1,1c00b2bc <__truncdfsf2+0x162>
1c00b322:	bfed                	j	1c00b31c <__truncdfsf2+0x1c2>
1c00b324:	4781                	li	a5,0
1c00b326:	00080c63          	beqz	a6,1c00b33e <__truncdfsf2+0x1e4>
1c00b32a:	00383763          	p.bneimm	a6,3,1c00b338 <__truncdfsf2+0x1de>
1c00b32e:	c981                	beqz	a1,1c00b33e <__truncdfsf2+0x1e4>
1c00b330:	57fd                	li	a5,-1
1c00b332:	0fe00693          	li	a3,254
1c00b336:	a021                	j	1c00b33e <__truncdfsf2+0x1e4>
1c00b338:	fe283ce3          	p.bneimm	a6,2,1c00b330 <__truncdfsf2+0x1d6>
1c00b33c:	d9f5                	beqz	a1,1c00b330 <__truncdfsf2+0x1d6>
1c00b33e:	00576713          	ori	a4,a4,5
1c00b342:	b545                	j	1c00b1e2 <__truncdfsf2+0x88>

1c00b344 <main>:
1c00b344:	1c001537          	lui	a0,0x1c001
1c00b348:	711d                	addi	sp,sp,-96
1c00b34a:	db050513          	addi	a0,a0,-592 # 1c000db0 <__clz_tab+0x100>
1c00b34e:	ce86                	sw	ra,92(sp)
1c00b350:	673000ef          	jal	ra,1c00c1c2 <puts>
1c00b354:	0808                	addi	a0,sp,16
1c00b356:	7cf020ef          	jal	ra,1c00e324 <pi_cluster_conf_init>
1c00b35a:	0048                	addi	a0,sp,4
1c00b35c:	080c                	addi	a1,sp,16
1c00b35e:	38b020ef          	jal	ra,1c00dee8 <pi_open_from_conf>
1c00b362:	0048                	addi	a0,sp,4
1c00b364:	7cb020ef          	jal	ra,1c00e32e <pi_cluster_open>
1c00b368:	c509                	beqz	a0,1c00b372 <main+0x2e>
1c00b36a:	40f6                	lw	ra,92(sp)
1c00b36c:	557d                	li	a0,-1
1c00b36e:	6125                	addi	sp,sp,96
1c00b370:	8082                	ret
1c00b372:	1c001537          	lui	a0,0x1c001
1c00b376:	dd450513          	addi	a0,a0,-556 # 1c000dd4 <__clz_tab+0x124>
1c00b37a:	649000ef          	jal	ra,1c00c1c2 <puts>
1c00b37e:	1c00b7b7          	lui	a5,0x1c00b
1c00b382:	3ba78793          	addi	a5,a5,954 # 1c00b3ba <net_step>
1c00b386:	102c                	addi	a1,sp,40
1c00b388:	d43e                	sw	a5,40(sp)
1c00b38a:	0048                	addi	a0,sp,4
1c00b38c:	47a1                	li	a5,8
1c00b38e:	de3e                	sw	a5,60(sp)
1c00b390:	d602                	sw	zero,44(sp)
1c00b392:	d802                	sw	zero,48(sp)
1c00b394:	da02                	sw	zero,52(sp)
1c00b396:	186030ef          	jal	ra,1c00e51c <pi_cluster_send_task_to_cl>
1c00b39a:	1c001537          	lui	a0,0x1c001
1c00b39e:	e0050513          	addi	a0,a0,-512 # 1c000e00 <__clz_tab+0x150>
1c00b3a2:	621000ef          	jal	ra,1c00c1c2 <puts>
1c00b3a6:	0048                	addi	a0,sp,4
1c00b3a8:	0f2030ef          	jal	ra,1c00e49a <pi_cluster_close>
1c00b3ac:	1c0027b7          	lui	a5,0x1c002
1c00b3b0:	4501                	li	a0,0
1c00b3b2:	9c07a823          	sw	zero,-1584(a5) # 1c0019d0 <pos_kernel_pmsis_exit_value>
1c00b3b6:	6a7000ef          	jal	ra,1c00c25c <exit>

1c00b3ba <net_step>:
1c00b3ba:	1c001537          	lui	a0,0x1c001
1c00b3be:	7119                	addi	sp,sp,-128
1c00b3c0:	e3850513          	addi	a0,a0,-456 # 1c000e38 <__clz_tab+0x188>
1c00b3c4:	de86                	sw	ra,124(sp)
1c00b3c6:	dca2                	sw	s0,120(sp)
1c00b3c8:	daa6                	sw	s1,116(sp)
1c00b3ca:	d8ca                	sw	s2,112(sp)
1c00b3cc:	d6ce                	sw	s3,108(sp)
1c00b3ce:	d4d2                	sw	s4,104(sp)
1c00b3d0:	d2d6                	sw	s5,100(sp)
1c00b3d2:	d0da                	sw	s6,96(sp)
1c00b3d4:	cede                	sw	s7,92(sp)
1c00b3d6:	cce2                	sw	s8,88(sp)
1c00b3d8:	cae6                	sw	s9,84(sp)
1c00b3da:	c8ea                	sw	s10,80(sp)
1c00b3dc:	c6ee                	sw	s11,76(sp)
1c00b3de:	5e5000ef          	jal	ra,1c00c1c2 <puts>
1c00b3e2:	000317b7          	lui	a5,0x31
1c00b3e6:	07dd                	addi	a5,a5,23
1c00b3e8:	cc079073          	csrw	0xcc0,a5
1c00b3ec:	f1402573          	csrr	a0,mhartid
1c00b3f0:	47fd                	li	a5,31
1c00b3f2:	d42a                	sw	a0,40(sp)
1c00b3f4:	ca551733          	p.extractu	a4,a0,5,5
1c00b3f8:	70f70263          	beq	a4,a5,1c00bafc <net_step+0x742>
1c00b3fc:	102007b7          	lui	a5,0x10200
1c00b400:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b404:	0007a023          	sw	zero,0(a5)
1c00b408:	4781                	li	a5,0
1c00b40a:	cc179073          	csrw	0xcc1,a5
1c00b40e:	85aa                	mv	a1,a0
1c00b410:	ca5597b3          	p.extractu	a5,a1,5,5
1c00b414:	477d                	li	a4,31
1c00b416:	6ce78963          	beq	a5,a4,1c00bae8 <net_step+0x72e>
1c00b41a:	102007b7          	lui	a5,0x10200
1c00b41e:	4705                	li	a4,1
1c00b420:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b424:	02e7a023          	sw	a4,32(a5)
1c00b428:	4781                	li	a5,0
1c00b42a:	79f79073          	csrw	pccr31,a5
1c00b42e:	86ae                	mv	a3,a1
1c00b430:	ca5697b3          	p.extractu	a5,a3,5,5
1c00b434:	477d                	li	a4,31
1c00b436:	6ae78063          	beq	a5,a4,1c00bad6 <net_step+0x71c>
1c00b43a:	102007b7          	lui	a5,0x10200
1c00b43e:	4705                	li	a4,1
1c00b440:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b444:	00e7ac23          	sw	a4,24(a5)
1c00b448:	478d                	li	a5,3
1c00b44a:	cc179073          	csrw	0xcc1,a5
1c00b44e:	10000d37          	lui	s10,0x10000
1c00b452:	01cd0c13          	addi	s8,s10,28 # 1000001c <gm_angles>
1c00b456:	320c0493          	addi	s1,s8,800
1c00b45a:	640c0c93          	addi	s9,s8,1600
1c00b45e:	89e6                	mv	s3,s9
1c00b460:	8926                	mv	s2,s1
1c00b462:	01cd0d13          	addi	s10,s10,28
1c00b466:	004d240b          	p.lw	s0,4(s10!)
1c00b46a:	8522                	mv	a0,s0
1c00b46c:	c31fc0ef          	jal	ra,1c00809c <cosf>
1c00b470:	00a9222b          	p.sw	a0,4(s2!)
1c00b474:	8522                	mv	a0,s0
1c00b476:	c83fc0ef          	jal	ra,1c0080f8 <sinf>
1c00b47a:	00a9a22b          	p.sw	a0,4(s3!)
1c00b47e:	fe9d14e3          	bne	s10,s1,1c00b466 <net_step+0xac>
1c00b482:	5722                	lw	a4,40(sp)
1c00b484:	ca5717b3          	p.extractu	a5,a4,5,5
1c00b488:	477d                	li	a4,31
1c00b48a:	62e78463          	beq	a5,a4,1c00bab2 <net_step+0x6f8>
1c00b48e:	102007b7          	lui	a5,0x10200
1c00b492:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b496:	0007a023          	sw	zero,0(a5)
1c00b49a:	4781                	li	a5,0
1c00b49c:	cc179073          	csrw	0xcc1,a5
1c00b4a0:	5822                	lw	a6,40(sp)
1c00b4a2:	477d                	li	a4,31
1c00b4a4:	ca5817b3          	p.extractu	a5,a6,5,5
1c00b4a8:	62e78263          	beq	a5,a4,1c00bacc <net_step+0x712>
1c00b4ac:	102007b7          	lui	a5,0x10200
1c00b4b0:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b4b4:	0087a783          	lw	a5,8(a5)
1c00b4b8:	1c002537          	lui	a0,0x1c002
1c00b4bc:	9b450513          	addi	a0,a0,-1612 # 1c0019b4 <_cycles>
1c00b4c0:	c11c                	sw	a5,0(a0)
1c00b4c2:	781027f3          	csrr	a5,pccr1
1c00b4c6:	1c0026b7          	lui	a3,0x1c002
1c00b4ca:	9bc68693          	addi	a3,a3,-1604 # 1c0019bc <_instr>
1c00b4ce:	c29c                	sw	a5,0(a3)
1c00b4d0:	780027f3          	csrr	a5,pccr0
1c00b4d4:	1c002837          	lui	a6,0x1c002
1c00b4d8:	9b080813          	addi	a6,a6,-1616 # 1c0019b0 <_edata>
1c00b4dc:	00f82023          	sw	a5,0(a6)
1c00b4e0:	78c027f3          	csrr	a5,pccr12
1c00b4e4:	1c0025b7          	lui	a1,0x1c002
1c00b4e8:	9c058593          	addi	a1,a1,-1600 # 1c0019c0 <_ldext>
1c00b4ec:	c19c                	sw	a5,0(a1)
1c00b4ee:	790027f3          	csrr	a5,pccr16
1c00b4f2:	1c002737          	lui	a4,0x1c002
1c00b4f6:	9c870713          	addi	a4,a4,-1592 # 1c0019c8 <_tcdmcont>
1c00b4fa:	c31c                	sw	a5,0(a4)
1c00b4fc:	782027f3          	csrr	a5,pccr2
1c00b500:	1c002537          	lui	a0,0x1c002
1c00b504:	9c450513          	addi	a0,a0,-1596 # 1c0019c4 <_ldstall>
1c00b508:	c11c                	sw	a5,0(a0)
1c00b50a:	78402773          	csrr	a4,pccr4
1c00b50e:	1c0026b7          	lui	a3,0x1c002
1c00b512:	9b868693          	addi	a3,a3,-1608 # 1c0019b8 <_imiss>
1c00b516:	c298                	sw	a4,0(a3)
1c00b518:	1c002737          	lui	a4,0x1c002
1c00b51c:	9cc70713          	addi	a4,a4,-1588 # 1c0019cc <id>
1c00b520:	f14027f3          	csrr	a5,mhartid
1c00b524:	4529                	li	a0,10
1c00b526:	f457b7b3          	p.bclr	a5,a5,26,5
1c00b52a:	c31c                	sw	a5,0(a4)
1c00b52c:	509000ef          	jal	ra,1c00c234 <putchar>
1c00b530:	1c0027b7          	lui	a5,0x1c002
1c00b534:	1c002837          	lui	a6,0x1c002
1c00b538:	9b478793          	addi	a5,a5,-1612 # 1c0019b4 <_cycles>
1c00b53c:	9cc80813          	addi	a6,a6,-1588 # 1c0019cc <id>
1c00b540:	4390                	lw	a2,0(a5)
1c00b542:	00082583          	lw	a1,0(a6)
1c00b546:	1c0016b7          	lui	a3,0x1c001
1c00b54a:	e6068513          	addi	a0,a3,-416 # 1c000e60 <__clz_tab+0x1b0>
1c00b54e:	543000ef          	jal	ra,1c00c290 <printf>
1c00b552:	1c002737          	lui	a4,0x1c002
1c00b556:	1c0027b7          	lui	a5,0x1c002
1c00b55a:	9cc78793          	addi	a5,a5,-1588 # 1c0019cc <id>
1c00b55e:	9bc70713          	addi	a4,a4,-1604 # 1c0019bc <_instr>
1c00b562:	438c                	lw	a1,0(a5)
1c00b564:	4310                	lw	a2,0(a4)
1c00b566:	1c001837          	lui	a6,0x1c001
1c00b56a:	e7480513          	addi	a0,a6,-396 # 1c000e74 <__clz_tab+0x1c4>
1c00b56e:	523000ef          	jal	ra,1c00c290 <printf>
1c00b572:	1c002537          	lui	a0,0x1c002
1c00b576:	1c0026b7          	lui	a3,0x1c002
1c00b57a:	9cc68693          	addi	a3,a3,-1588 # 1c0019cc <id>
1c00b57e:	9b050513          	addi	a0,a0,-1616 # 1c0019b0 <_edata>
1c00b582:	4110                	lw	a2,0(a0)
1c00b584:	428c                	lw	a1,0(a3)
1c00b586:	1c001737          	lui	a4,0x1c001
1c00b58a:	e8870513          	addi	a0,a4,-376 # 1c000e88 <__clz_tab+0x1d8>
1c00b58e:	503000ef          	jal	ra,1c00c290 <printf>
1c00b592:	1c0027b7          	lui	a5,0x1c002
1c00b596:	1c002837          	lui	a6,0x1c002
1c00b59a:	9c078793          	addi	a5,a5,-1600 # 1c0019c0 <_ldext>
1c00b59e:	9cc80813          	addi	a6,a6,-1588 # 1c0019cc <id>
1c00b5a2:	4390                	lw	a2,0(a5)
1c00b5a4:	00082583          	lw	a1,0(a6)
1c00b5a8:	1c0016b7          	lui	a3,0x1c001
1c00b5ac:	ea468513          	addi	a0,a3,-348 # 1c000ea4 <__clz_tab+0x1f4>
1c00b5b0:	4e1000ef          	jal	ra,1c00c290 <printf>
1c00b5b4:	1c002737          	lui	a4,0x1c002
1c00b5b8:	1c0027b7          	lui	a5,0x1c002
1c00b5bc:	9cc78793          	addi	a5,a5,-1588 # 1c0019cc <id>
1c00b5c0:	9c870713          	addi	a4,a4,-1592 # 1c0019c8 <_tcdmcont>
1c00b5c4:	438c                	lw	a1,0(a5)
1c00b5c6:	4310                	lw	a2,0(a4)
1c00b5c8:	1c001837          	lui	a6,0x1c001
1c00b5cc:	ebc80513          	addi	a0,a6,-324 # 1c000ebc <__clz_tab+0x20c>
1c00b5d0:	4c1000ef          	jal	ra,1c00c290 <printf>
1c00b5d4:	1c002537          	lui	a0,0x1c002
1c00b5d8:	1c0026b7          	lui	a3,0x1c002
1c00b5dc:	9cc68693          	addi	a3,a3,-1588 # 1c0019cc <id>
1c00b5e0:	9c450513          	addi	a0,a0,-1596 # 1c0019c4 <_ldstall>
1c00b5e4:	4110                	lw	a2,0(a0)
1c00b5e6:	428c                	lw	a1,0(a3)
1c00b5e8:	1c001737          	lui	a4,0x1c001
1c00b5ec:	ed470513          	addi	a0,a4,-300 # 1c000ed4 <__clz_tab+0x224>
1c00b5f0:	4a1000ef          	jal	ra,1c00c290 <printf>
1c00b5f4:	1c0027b7          	lui	a5,0x1c002
1c00b5f8:	1c002837          	lui	a6,0x1c002
1c00b5fc:	9b878793          	addi	a5,a5,-1608 # 1c0019b8 <_imiss>
1c00b600:	9cc80813          	addi	a6,a6,-1588 # 1c0019cc <id>
1c00b604:	4390                	lw	a2,0(a5)
1c00b606:	00082583          	lw	a1,0(a6)
1c00b60a:	1c0016b7          	lui	a3,0x1c001
1c00b60e:	eec68513          	addi	a0,a3,-276 # 1c000eec <__clz_tab+0x23c>
1c00b612:	47f000ef          	jal	ra,1c00c290 <printf>
1c00b616:	1c001837          	lui	a6,0x1c001
1c00b61a:	f0880713          	addi	a4,a6,-248 # 1c000f08 <__clz_tab+0x258>
1c00b61e:	1c001837          	lui	a6,0x1c001
1c00b622:	1c001537          	lui	a0,0x1c001
1c00b626:	f2080593          	addi	a1,a6,-224 # 1c000f20 <__clz_tab+0x270>
1c00b62a:	1c0017b7          	lui	a5,0x1c001
1c00b62e:	1c001837          	lui	a6,0x1c001
1c00b632:	1c0016b7          	lui	a3,0x1c001
1c00b636:	f1850613          	addi	a2,a0,-232 # 1c000f18 <__clz_tab+0x268>
1c00b63a:	f0078793          	addi	a5,a5,-256 # 1c000f00 <__clz_tab+0x250>
1c00b63e:	f1068693          	addi	a3,a3,-240 # 1c000f10 <__clz_tab+0x260>
1c00b642:	f2880513          	addi	a0,a6,-216 # 1c000f28 <__clz_tab+0x278>
1c00b646:	44b000ef          	jal	ra,1c00c290 <printf>
1c00b64a:	1c010db7          	lui	s11,0x1c010
1c00b64e:	1c010ab7          	lui	s5,0x1c010
1c00b652:	1c010b37          	lui	s6,0x1c010
1c00b656:	150d8413          	addi	s0,s11,336 # 1c010150 <gm_cos>
1c00b65a:	470a8a93          	addi	s5,s5,1136 # 1c010470 <gm_sin>
1c00b65e:	8326                	mv	t1,s1
1c00b660:	470b0b13          	addi	s6,s6,1136 # 1c010470 <gm_sin>
1c00b664:	150d8d93          	addi	s11,s11,336
1c00b668:	d662                	sw	s8,44(sp)
1c00b66a:	d866                	sw	s9,48(sp)
1c00b66c:	da56                	sw	s5,52(sp)
1c00b66e:	dc26                	sw	s1,56(sp)
1c00b670:	8ae2                	mv	s5,s8
1c00b672:	8a66                	mv	s4,s9
1c00b674:	de22                	sw	s0,60(sp)
1c00b676:	84da                	mv	s1,s6
1c00b678:	d26a                	sw	s10,36(sp)
1c00b67a:	8c6e                	mv	s8,s11
1c00b67c:	8c9a                	mv	s9,t1
1c00b67e:	004a270b          	p.lw	a4,4(s4!)
1c00b682:	004ca50b          	p.lw	a0,4(s9!)
1c00b686:	004c298b          	p.lw	s3,4(s8!)
1c00b68a:	004aa90b          	p.lw	s2,4(s5!)
1c00b68e:	ce3a                	sw	a4,28(sp)
1c00b690:	341d                	jal	1c00b0b6 <__extendsfdf2>
1c00b692:	8d2a                	mv	s10,a0
1c00b694:	854e                	mv	a0,s3
1c00b696:	8dae                	mv	s11,a1
1c00b698:	0044a40b          	p.lw	s0,4(s1!) # 10004 <_l1_preload_size+0xf684>
1c00b69c:	3c29                	jal	1c00b0b6 <__extendsfdf2>
1c00b69e:	8b2a                	mv	s6,a0
1c00b6a0:	854a                	mv	a0,s2
1c00b6a2:	8bae                	mv	s7,a1
1c00b6a4:	3c09                	jal	1c00b0b6 <__extendsfdf2>
1c00b6a6:	4772                	lw	a4,28(sp)
1c00b6a8:	892a                	mv	s2,a0
1c00b6aa:	89ae                	mv	s3,a1
1c00b6ac:	853a                	mv	a0,a4
1c00b6ae:	3421                	jal	1c00b0b6 <__extendsfdf2>
1c00b6b0:	c42a                	sw	a0,8(sp)
1c00b6b2:	8522                	mv	a0,s0
1c00b6b4:	c62e                	sw	a1,12(sp)
1c00b6b6:	3401                	jal	1c00b0b6 <__extendsfdf2>
1c00b6b8:	c22e                	sw	a1,4(sp)
1c00b6ba:	1c0015b7          	lui	a1,0x1c001
1c00b6be:	c02a                	sw	a0,0(sp)
1c00b6c0:	86ce                	mv	a3,s3
1c00b6c2:	886a                	mv	a6,s10
1c00b6c4:	88ee                	mv	a7,s11
1c00b6c6:	875a                	mv	a4,s6
1c00b6c8:	87de                	mv	a5,s7
1c00b6ca:	864a                	mv	a2,s2
1c00b6cc:	f4458513          	addi	a0,a1,-188 # 1c000f44 <__clz_tab+0x294>
1c00b6d0:	3c1000ef          	jal	ra,1c00c290 <printf>
1c00b6d4:	5692                	lw	a3,36(sp)
1c00b6d6:	fb5694e3          	bne	a3,s5,1c00b67e <net_step+0x2c4>
1c00b6da:	1c001537          	lui	a0,0x1c001
1c00b6de:	f6050513          	addi	a0,a0,-160 # 1c000f60 <__clz_tab+0x2b0>
1c00b6e2:	5c32                	lw	s8,44(sp)
1c00b6e4:	5cc2                	lw	s9,48(sp)
1c00b6e6:	54e2                	lw	s1,56(sp)
1c00b6e8:	5472                	lw	s0,60(sp)
1c00b6ea:	8a56                	mv	s4,s5
1c00b6ec:	5ad2                	lw	s5,52(sp)
1c00b6ee:	2d5000ef          	jal	ra,1c00c1c2 <puts>
1c00b6f2:	000317b7          	lui	a5,0x31
1c00b6f6:	07dd                	addi	a5,a5,23
1c00b6f8:	cc079073          	csrw	0xcc0,a5
1c00b6fc:	5822                	lw	a6,40(sp)
1c00b6fe:	47fd                	li	a5,31
1c00b700:	ca581733          	p.extractu	a4,a6,5,5
1c00b704:	34f70163          	beq	a4,a5,1c00ba46 <net_step+0x68c>
1c00b708:	102007b7          	lui	a5,0x10200
1c00b70c:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b710:	0007a023          	sw	zero,0(a5)
1c00b714:	4781                	li	a5,0
1c00b716:	cc179073          	csrw	0xcc1,a5
1c00b71a:	8542                	mv	a0,a6
1c00b71c:	ca5517b3          	p.extractu	a5,a0,5,5
1c00b720:	477d                	li	a4,31
1c00b722:	32e78f63          	beq	a5,a4,1c00ba60 <net_step+0x6a6>
1c00b726:	102007b7          	lui	a5,0x10200
1c00b72a:	4705                	li	a4,1
1c00b72c:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b730:	02e7a023          	sw	a4,32(a5)
1c00b734:	4781                	li	a5,0
1c00b736:	79f79073          	csrw	pccr31,a5
1c00b73a:	85aa                	mv	a1,a0
1c00b73c:	ca5597b3          	p.extractu	a5,a1,5,5
1c00b740:	477d                	li	a4,31
1c00b742:	32e78d63          	beq	a5,a4,1c00ba7c <net_step+0x6c2>
1c00b746:	102007b7          	lui	a5,0x10200
1c00b74a:	4705                	li	a4,1
1c00b74c:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b750:	00e7ac23          	sw	a4,24(a5)
1c00b754:	478d                	li	a5,3
1c00b756:	cc179073          	csrw	0xcc1,a5
1c00b75a:	8b66                	mv	s6,s9
1c00b75c:	89a6                	mv	s3,s1
1c00b75e:	8962                	mv	s2,s8
1c00b760:	0049250b          	p.lw	a0,4(s2!)
1c00b764:	865a                	mv	a2,s6
1c00b766:	85ce                	mv	a1,s3
1c00b768:	265d                	jal	1c00bb0e <cordic_cos_sin_fp32>
1c00b76a:	0991                	addi	s3,s3,4
1c00b76c:	0b11                	addi	s6,s6,4
1c00b76e:	ff4919e3          	bne	s2,s4,1c00b760 <net_step+0x3a6>
1c00b772:	56a2                	lw	a3,40(sp)
1c00b774:	477d                	li	a4,31
1c00b776:	ca5697b3          	p.extractu	a5,a3,5,5
1c00b77a:	30e78a63          	beq	a5,a4,1c00ba8e <net_step+0x6d4>
1c00b77e:	102007b7          	lui	a5,0x10200
1c00b782:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b786:	0007a023          	sw	zero,0(a5)
1c00b78a:	4781                	li	a5,0
1c00b78c:	cc179073          	csrw	0xcc1,a5
1c00b790:	8736                	mv	a4,a3
1c00b792:	ca5717b3          	p.extractu	a5,a4,5,5
1c00b796:	477d                	li	a4,31
1c00b798:	30e78863          	beq	a5,a4,1c00baa8 <net_step+0x6ee>
1c00b79c:	102007b7          	lui	a5,0x10200
1c00b7a0:	40078793          	addi	a5,a5,1024 # 10200400 <__l1_end+0x1ffa7c>
1c00b7a4:	0087a783          	lw	a5,8(a5)
1c00b7a8:	1c002837          	lui	a6,0x1c002
1c00b7ac:	9b480813          	addi	a6,a6,-1612 # 1c0019b4 <_cycles>
1c00b7b0:	00f82023          	sw	a5,0(a6)
1c00b7b4:	781027f3          	csrr	a5,pccr1
1c00b7b8:	1c0025b7          	lui	a1,0x1c002
1c00b7bc:	9bc58593          	addi	a1,a1,-1604 # 1c0019bc <_instr>
1c00b7c0:	c19c                	sw	a5,0(a1)
1c00b7c2:	780027f3          	csrr	a5,pccr0
1c00b7c6:	1c002737          	lui	a4,0x1c002
1c00b7ca:	9b070713          	addi	a4,a4,-1616 # 1c0019b0 <_edata>
1c00b7ce:	c31c                	sw	a5,0(a4)
1c00b7d0:	78c027f3          	csrr	a5,pccr12
1c00b7d4:	1c002537          	lui	a0,0x1c002
1c00b7d8:	9c050513          	addi	a0,a0,-1600 # 1c0019c0 <_ldext>
1c00b7dc:	c11c                	sw	a5,0(a0)
1c00b7de:	790027f3          	csrr	a5,pccr16
1c00b7e2:	1c0026b7          	lui	a3,0x1c002
1c00b7e6:	9c868693          	addi	a3,a3,-1592 # 1c0019c8 <_tcdmcont>
1c00b7ea:	c29c                	sw	a5,0(a3)
1c00b7ec:	782027f3          	csrr	a5,pccr2
1c00b7f0:	1c002837          	lui	a6,0x1c002
1c00b7f4:	9c480813          	addi	a6,a6,-1596 # 1c0019c4 <_ldstall>
1c00b7f8:	00f82023          	sw	a5,0(a6)
1c00b7fc:	78402773          	csrr	a4,pccr4
1c00b800:	1c0025b7          	lui	a1,0x1c002
1c00b804:	1c0026b7          	lui	a3,0x1c002
1c00b808:	9cc68693          	addi	a3,a3,-1588 # 1c0019cc <id>
1c00b80c:	9b858593          	addi	a1,a1,-1608 # 1c0019b8 <_imiss>
1c00b810:	f14027f3          	csrr	a5,mhartid
1c00b814:	4529                	li	a0,10
1c00b816:	f457b7b3          	p.bclr	a5,a5,26,5
1c00b81a:	c29c                	sw	a5,0(a3)
1c00b81c:	c198                	sw	a4,0(a1)
1c00b81e:	217000ef          	jal	ra,1c00c234 <putchar>
1c00b822:	1c002737          	lui	a4,0x1c002
1c00b826:	1c0027b7          	lui	a5,0x1c002
1c00b82a:	9cc78793          	addi	a5,a5,-1588 # 1c0019cc <id>
1c00b82e:	9b470713          	addi	a4,a4,-1612 # 1c0019b4 <_cycles>
1c00b832:	438c                	lw	a1,0(a5)
1c00b834:	4310                	lw	a2,0(a4)
1c00b836:	1c001837          	lui	a6,0x1c001
1c00b83a:	e6080513          	addi	a0,a6,-416 # 1c000e60 <__clz_tab+0x1b0>
1c00b83e:	253000ef          	jal	ra,1c00c290 <printf>
1c00b842:	1c002537          	lui	a0,0x1c002
1c00b846:	1c0026b7          	lui	a3,0x1c002
1c00b84a:	9cc68693          	addi	a3,a3,-1588 # 1c0019cc <id>
1c00b84e:	9bc50513          	addi	a0,a0,-1604 # 1c0019bc <_instr>
1c00b852:	4110                	lw	a2,0(a0)
1c00b854:	428c                	lw	a1,0(a3)
1c00b856:	1c001737          	lui	a4,0x1c001
1c00b85a:	e7470513          	addi	a0,a4,-396 # 1c000e74 <__clz_tab+0x1c4>
1c00b85e:	233000ef          	jal	ra,1c00c290 <printf>
1c00b862:	1c0027b7          	lui	a5,0x1c002
1c00b866:	1c002837          	lui	a6,0x1c002
1c00b86a:	9b078793          	addi	a5,a5,-1616 # 1c0019b0 <_edata>
1c00b86e:	9cc80813          	addi	a6,a6,-1588 # 1c0019cc <id>
1c00b872:	4390                	lw	a2,0(a5)
1c00b874:	00082583          	lw	a1,0(a6)
1c00b878:	1c0016b7          	lui	a3,0x1c001
1c00b87c:	e8868513          	addi	a0,a3,-376 # 1c000e88 <__clz_tab+0x1d8>
1c00b880:	211000ef          	jal	ra,1c00c290 <printf>
1c00b884:	1c002737          	lui	a4,0x1c002
1c00b888:	1c0027b7          	lui	a5,0x1c002
1c00b88c:	9cc78793          	addi	a5,a5,-1588 # 1c0019cc <id>
1c00b890:	9c070713          	addi	a4,a4,-1600 # 1c0019c0 <_ldext>
1c00b894:	438c                	lw	a1,0(a5)
1c00b896:	4310                	lw	a2,0(a4)
1c00b898:	1c001837          	lui	a6,0x1c001
1c00b89c:	ea480513          	addi	a0,a6,-348 # 1c000ea4 <__clz_tab+0x1f4>
1c00b8a0:	1f1000ef          	jal	ra,1c00c290 <printf>
1c00b8a4:	1c002537          	lui	a0,0x1c002
1c00b8a8:	1c0026b7          	lui	a3,0x1c002
1c00b8ac:	9cc68693          	addi	a3,a3,-1588 # 1c0019cc <id>
1c00b8b0:	9c850513          	addi	a0,a0,-1592 # 1c0019c8 <_tcdmcont>
1c00b8b4:	4110                	lw	a2,0(a0)
1c00b8b6:	428c                	lw	a1,0(a3)
1c00b8b8:	1c001737          	lui	a4,0x1c001
1c00b8bc:	ebc70513          	addi	a0,a4,-324 # 1c000ebc <__clz_tab+0x20c>
1c00b8c0:	1d1000ef          	jal	ra,1c00c290 <printf>
1c00b8c4:	1c0027b7          	lui	a5,0x1c002
1c00b8c8:	1c002837          	lui	a6,0x1c002
1c00b8cc:	9c478793          	addi	a5,a5,-1596 # 1c0019c4 <_ldstall>
1c00b8d0:	9cc80813          	addi	a6,a6,-1588 # 1c0019cc <id>
1c00b8d4:	4390                	lw	a2,0(a5)
1c00b8d6:	00082583          	lw	a1,0(a6)
1c00b8da:	1c0016b7          	lui	a3,0x1c001
1c00b8de:	ed468513          	addi	a0,a3,-300 # 1c000ed4 <__clz_tab+0x224>
1c00b8e2:	1af000ef          	jal	ra,1c00c290 <printf>
1c00b8e6:	1c002737          	lui	a4,0x1c002
1c00b8ea:	1c0027b7          	lui	a5,0x1c002
1c00b8ee:	9b870713          	addi	a4,a4,-1608 # 1c0019b8 <_imiss>
1c00b8f2:	9cc78793          	addi	a5,a5,-1588 # 1c0019cc <id>
1c00b8f6:	4310                	lw	a2,0(a4)
1c00b8f8:	438c                	lw	a1,0(a5)
1c00b8fa:	1c001837          	lui	a6,0x1c001
1c00b8fe:	eec80513          	addi	a0,a6,-276 # 1c000eec <__clz_tab+0x23c>
1c00b902:	18f000ef          	jal	ra,1c00c290 <printf>
1c00b906:	1c0015b7          	lui	a1,0x1c001
1c00b90a:	1c001837          	lui	a6,0x1c001
1c00b90e:	00000913          	li	s2,0
1c00b912:	1c001537          	lui	a0,0x1c001
1c00b916:	f1858613          	addi	a2,a1,-232 # 1c000f18 <__clz_tab+0x268>
1c00b91a:	1c0017b7          	lui	a5,0x1c001
1c00b91e:	f2080593          	addi	a1,a6,-224 # 1c000f20 <__clz_tab+0x270>
1c00b922:	1c0016b7          	lui	a3,0x1c001
1c00b926:	1c001837          	lui	a6,0x1c001
1c00b92a:	f0850713          	addi	a4,a0,-248 # 1c000f08 <__clz_tab+0x258>
1c00b92e:	f8078793          	addi	a5,a5,-128 # 1c000f80 <__clz_tab+0x2d0>
1c00b932:	f8c68693          	addi	a3,a3,-116 # 1c000f8c <__clz_tab+0x2dc>
1c00b936:	f2880513          	addi	a0,a6,-216 # 1c000f28 <__clz_tab+0x278>
1c00b93a:	157000ef          	jal	ra,1c00c290 <printf>
1c00b93e:	8a4a                	mv	s4,s2
1c00b940:	834a                	mv	t1,s2
1c00b942:	8e4a                	mv	t3,s2
1c00b944:	a88d                	j	1c00b9b6 <net_step+0x5fc>
1c00b946:	00e37353          	fadd.s	t1,t1,a4
1c00b94a:	a0fa1753          	flt.s	a4,s4,a5
1c00b94e:	cb41                	beqz	a4,1c00b9de <net_step+0x624>
1c00b950:	00fe7e53          	fadd.s	t3,t3,a5
1c00b954:	004c2d8b          	p.lw	s11,4(s8!)
1c00b958:	cc1a                	sw	t1,24(sp)
1c00b95a:	ca72                	sw	t3,20(sp)
1c00b95c:	f5aff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00b960:	892a                	mv	s2,a0
1c00b962:	856a                	mv	a0,s10
1c00b964:	89ae                	mv	s3,a1
1c00b966:	f50ff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00b96a:	872a                	mv	a4,a0
1c00b96c:	856e                	mv	a0,s11
1c00b96e:	ce3a                	sw	a4,28(sp)
1c00b970:	d02e                	sw	a1,32(sp)
1c00b972:	f44ff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00b976:	8d2a                	mv	s10,a0
1c00b978:	855e                	mv	a0,s7
1c00b97a:	8dae                	mv	s11,a1
1c00b97c:	f3aff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00b980:	c42a                	sw	a0,8(sp)
1c00b982:	855a                	mv	a0,s6
1c00b984:	c62e                	sw	a1,12(sp)
1c00b986:	f30ff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00b98a:	4772                	lw	a4,28(sp)
1c00b98c:	5782                	lw	a5,32(sp)
1c00b98e:	c22e                	sw	a1,4(sp)
1c00b990:	1c0015b7          	lui	a1,0x1c001
1c00b994:	c02a                	sw	a0,0(sp)
1c00b996:	86ee                	mv	a3,s11
1c00b998:	884a                	mv	a6,s2
1c00b99a:	88ce                	mv	a7,s3
1c00b99c:	866a                	mv	a2,s10
1c00b99e:	f4458513          	addi	a0,a1,-188 # 1c000f44 <__clz_tab+0x294>
1c00b9a2:	0ef000ef          	jal	ra,1c00c290 <printf>
1c00b9a6:	1c0106b7          	lui	a3,0x1c010
1c00b9aa:	47068693          	addi	a3,a3,1136 # 1c010470 <gm_sin>
1c00b9ae:	4362                	lw	t1,24(sp)
1c00b9b0:	4e52                	lw	t3,20(sp)
1c00b9b2:	02d40963          	beq	s0,a3,1c00b9e4 <net_step+0x62a>
1c00b9b6:	00442d0b          	p.lw	s10,4(s0!)
1c00b9ba:	0044a50b          	p.lw	a0,4(s1!)
1c00b9be:	004aab0b          	p.lw	s6,4(s5!)
1c00b9c2:	004cab8b          	p.lw	s7,4(s9!)
1c00b9c6:	08ad7753          	fsub.s	a4,s10,a0
1c00b9ca:	097b77d3          	fsub.s	a5,s6,s7
1c00b9ce:	a0ea16d3          	flt.s	a3,s4,a4
1c00b9d2:	fab5                	bnez	a3,1c00b946 <net_step+0x58c>
1c00b9d4:	08e37353          	fsub.s	t1,t1,a4
1c00b9d8:	a0fa1753          	flt.s	a4,s4,a5
1c00b9dc:	fb35                	bnez	a4,1c00b950 <net_step+0x596>
1c00b9de:	08fe7e53          	fsub.s	t3,t3,a5
1c00b9e2:	bf8d                	j	1c00b954 <net_step+0x59a>
1c00b9e4:	1c0017b7          	lui	a5,0x1c001
1c00b9e8:	e347a403          	lw	s0,-460(a5) # 1c000e34 <__clz_tab+0x184>
1c00b9ec:	8972                	mv	s2,t3
1c00b9ee:	18837553          	fdiv.s	a0,t1,s0
1c00b9f2:	ec4ff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00b9f6:	862a                	mv	a2,a0
1c00b9f8:	1c001537          	lui	a0,0x1c001
1c00b9fc:	86ae                	mv	a3,a1
1c00b9fe:	f9850513          	addi	a0,a0,-104 # 1c000f98 <__clz_tab+0x2e8>
1c00ba02:	08f000ef          	jal	ra,1c00c290 <printf>
1c00ba06:	18897553          	fdiv.s	a0,s2,s0
1c00ba0a:	eacff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00ba0e:	862a                	mv	a2,a0
1c00ba10:	1c001537          	lui	a0,0x1c001
1c00ba14:	86ae                	mv	a3,a1
1c00ba16:	fb450513          	addi	a0,a0,-76 # 1c000fb4 <__clz_tab+0x304>
1c00ba1a:	077000ef          	jal	ra,1c00c290 <printf>
1c00ba1e:	5466                	lw	s0,120(sp)
1c00ba20:	50f6                	lw	ra,124(sp)
1c00ba22:	54d6                	lw	s1,116(sp)
1c00ba24:	5946                	lw	s2,112(sp)
1c00ba26:	59b6                	lw	s3,108(sp)
1c00ba28:	5a26                	lw	s4,104(sp)
1c00ba2a:	5a96                	lw	s5,100(sp)
1c00ba2c:	5b06                	lw	s6,96(sp)
1c00ba2e:	4bf6                	lw	s7,92(sp)
1c00ba30:	4c66                	lw	s8,88(sp)
1c00ba32:	4cd6                	lw	s9,84(sp)
1c00ba34:	4d46                	lw	s10,80(sp)
1c00ba36:	4db6                	lw	s11,76(sp)
1c00ba38:	1c001537          	lui	a0,0x1c001
1c00ba3c:	fcc50513          	addi	a0,a0,-52 # 1c000fcc <__clz_tab+0x31c>
1c00ba40:	6109                	addi	sp,sp,128
1c00ba42:	7800006f          	j	1c00c1c2 <puts>
1c00ba46:	1a10b7b7          	lui	a5,0x1a10b
1c00ba4a:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa10a67c>
1c00ba4e:	4781                	li	a5,0
1c00ba50:	cc179073          	csrw	0xcc1,a5
1c00ba54:	8542                	mv	a0,a6
1c00ba56:	ca5517b3          	p.extractu	a5,a0,5,5
1c00ba5a:	477d                	li	a4,31
1c00ba5c:	cce795e3          	bne	a5,a4,1c00b726 <net_step+0x36c>
1c00ba60:	4785                	li	a5,1
1c00ba62:	1a10b737          	lui	a4,0x1a10b
1c00ba66:	02f72023          	sw	a5,32(a4) # 1a10b020 <__l1_end+0xa10a69c>
1c00ba6a:	4781                	li	a5,0
1c00ba6c:	79f79073          	csrw	pccr31,a5
1c00ba70:	85aa                	mv	a1,a0
1c00ba72:	ca5597b3          	p.extractu	a5,a1,5,5
1c00ba76:	477d                	li	a4,31
1c00ba78:	cce797e3          	bne	a5,a4,1c00b746 <net_step+0x38c>
1c00ba7c:	4785                	li	a5,1
1c00ba7e:	1a10b737          	lui	a4,0x1a10b
1c00ba82:	00f72c23          	sw	a5,24(a4) # 1a10b018 <__l1_end+0xa10a694>
1c00ba86:	478d                	li	a5,3
1c00ba88:	cc179073          	csrw	0xcc1,a5
1c00ba8c:	b1f9                	j	1c00b75a <net_step+0x3a0>
1c00ba8e:	1a10b7b7          	lui	a5,0x1a10b
1c00ba92:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa10a67c>
1c00ba96:	4781                	li	a5,0
1c00ba98:	cc179073          	csrw	0xcc1,a5
1c00ba9c:	8736                	mv	a4,a3
1c00ba9e:	ca5717b3          	p.extractu	a5,a4,5,5
1c00baa2:	477d                	li	a4,31
1c00baa4:	cee79ce3          	bne	a5,a4,1c00b79c <net_step+0x3e2>
1c00baa8:	1a10b7b7          	lui	a5,0x1a10b
1c00baac:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa10a684>
1c00bab0:	b9e5                	j	1c00b7a8 <net_step+0x3ee>
1c00bab2:	1a10b7b7          	lui	a5,0x1a10b
1c00bab6:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa10a67c>
1c00baba:	4781                	li	a5,0
1c00babc:	cc179073          	csrw	0xcc1,a5
1c00bac0:	5822                	lw	a6,40(sp)
1c00bac2:	477d                	li	a4,31
1c00bac4:	ca5817b3          	p.extractu	a5,a6,5,5
1c00bac8:	9ee792e3          	bne	a5,a4,1c00b4ac <net_step+0xf2>
1c00bacc:	1a10b7b7          	lui	a5,0x1a10b
1c00bad0:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa10a684>
1c00bad4:	b2d5                	j	1c00b4b8 <net_step+0xfe>
1c00bad6:	4785                	li	a5,1
1c00bad8:	1a10b737          	lui	a4,0x1a10b
1c00badc:	00f72c23          	sw	a5,24(a4) # 1a10b018 <__l1_end+0xa10a694>
1c00bae0:	478d                	li	a5,3
1c00bae2:	cc179073          	csrw	0xcc1,a5
1c00bae6:	b2a5                	j	1c00b44e <net_step+0x94>
1c00bae8:	4785                	li	a5,1
1c00baea:	1a10b737          	lui	a4,0x1a10b
1c00baee:	02f72023          	sw	a5,32(a4) # 1a10b020 <__l1_end+0xa10a69c>
1c00baf2:	4781                	li	a5,0
1c00baf4:	79f79073          	csrw	pccr31,a5
1c00baf8:	86ae                	mv	a3,a1
1c00bafa:	ba1d                	j	1c00b430 <net_step+0x76>
1c00bafc:	1a10b7b7          	lui	a5,0x1a10b
1c00bb00:	0007a023          	sw	zero,0(a5) # 1a10b000 <__l1_end+0xa10a67c>
1c00bb04:	4781                	li	a5,0
1c00bb06:	cc179073          	csrw	0xcc1,a5
1c00bb0a:	85aa                	mv	a1,a0
1c00bb0c:	b211                	j	1c00b410 <net_step+0x56>

1c00bb0e <cordic_cos_sin_fp32>:
1c00bb0e:	1101                	addi	sp,sp,-32
1c00bb10:	c64e                	sw	s3,12(sp)
1c00bb12:	1c0019b7          	lui	s3,0x1c001
1c00bb16:	ce06                	sw	ra,28(sp)
1c00bb18:	05098993          	addi	s3,s3,80 # 1c001050 <__clz_tab+0x3a0>
1c00bb1c:	cc22                	sw	s0,24(sp)
1c00bb1e:	ca26                	sw	s1,20(sp)
1c00bb20:	c84a                	sw	s2,16(sp)
1c00bb22:	c452                	sw	s4,8(sp)
1c00bb24:	c256                	sw	s5,4(sp)
1c00bb26:	c05a                	sw	s6,0(sp)
1c00bb28:	84b2                	mv	s1,a2
1c00bb2a:	892e                	mv	s2,a1
1c00bb2c:	d8aff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00bb30:	0009a603          	lw	a2,0(s3)
1c00bb34:	0049a683          	lw	a3,4(s3)
1c00bb38:	8a2a                	mv	s4,a0
1c00bb3a:	8aae                	mv	s5,a1
1c00bb3c:	f3dfd0ef          	jal	ra,1c009a78 <__divdf3>
1c00bb40:	c66ff0ef          	jal	ra,1c00afa6 <__fixdfsi>
1c00bb44:	d1aff0ef          	jal	ra,1c00b05e <__floatsidf>
1c00bb48:	0009a603          	lw	a2,0(s3)
1c00bb4c:	0049a683          	lw	a3,4(s3)
1c00bb50:	1c001b37          	lui	s6,0x1c001
1c00bb54:	058b0b13          	addi	s6,s6,88 # 1c001058 <__clz_tab+0x3a8>
1c00bb58:	e74fe0ef          	jal	ra,1c00a1cc <__muldf3>
1c00bb5c:	862a                	mv	a2,a0
1c00bb5e:	86ae                	mv	a3,a1
1c00bb60:	8552                	mv	a0,s4
1c00bb62:	85d6                	mv	a1,s5
1c00bb64:	bcffe0ef          	jal	ra,1c00a732 <__subdf3>
1c00bb68:	df2ff0ef          	jal	ra,1c00b15a <__truncdfsf2>
1c00bb6c:	842a                	mv	s0,a0
1c00bb6e:	d48ff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00bb72:	000b2603          	lw	a2,0(s6)
1c00bb76:	004b2683          	lw	a3,4(s6)
1c00bb7a:	8a2a                	mv	s4,a0
1c00bb7c:	8aae                	mv	s5,a1
1c00bb7e:	cfefe0ef          	jal	ra,1c00a07c <__gedf2>
1c00bb82:	2ea05c63          	blez	a0,1c00be7a <cordic_cos_sin_fp32+0x36c>
1c00bb86:	0009a603          	lw	a2,0(s3)
1c00bb8a:	0049a683          	lw	a3,4(s3)
1c00bb8e:	8552                	mv	a0,s4
1c00bb90:	85d6                	mv	a1,s5
1c00bb92:	ba1fe0ef          	jal	ra,1c00a732 <__subdf3>
1c00bb96:	dc4ff0ef          	jal	ra,1c00b15a <__truncdfsf2>
1c00bb9a:	842a                	mv	s0,a0
1c00bb9c:	d1aff0ef          	jal	ra,1c00b0b6 <__extendsfdf2>
1c00bba0:	8a2a                	mv	s4,a0
1c00bba2:	8aae                	mv	s5,a1
1c00bba4:	1c0017b7          	lui	a5,0x1c001
1c00bba8:	06878793          	addi	a5,a5,104 # 1c001068 <__clz_tab+0x3b8>
1c00bbac:	4390                	lw	a2,0(a5)
1c00bbae:	43d4                	lw	a3,4(a5)
1c00bbb0:	8552                	mv	a0,s4
1c00bbb2:	85d6                	mv	a1,s5
1c00bbb4:	cc8fe0ef          	jal	ra,1c00a07c <__gedf2>
1c00bbb8:	2ea05663          	blez	a0,1c00bea4 <cordic_cos_sin_fp32+0x396>
1c00bbbc:	000b2503          	lw	a0,0(s6)
1c00bbc0:	004b2583          	lw	a1,4(s6)
1c00bbc4:	86d6                	mv	a3,s5
1c00bbc6:	8652                	mv	a2,s4
1c00bbc8:	b6bfe0ef          	jal	ra,1c00a732 <__subdf3>
1c00bbcc:	d8eff0ef          	jal	ra,1c00b15a <__truncdfsf2>
1c00bbd0:	1c0017b7          	lui	a5,0x1c001
1c00bbd4:	842a                	mv	s0,a0
1c00bbd6:	fe07a683          	lw	a3,-32(a5) # 1c000fe0 <__clz_tab+0x330>
1c00bbda:	00000793          	li	a5,0
1c00bbde:	a08797d3          	flt.s	a5,a5,s0
1c00bbe2:	2e078e63          	beqz	a5,1c00bede <cordic_cos_sin_fp32+0x3d0>
1c00bbe6:	1c0017b7          	lui	a5,0x1c001
1c00bbea:	fec7a783          	lw	a5,-20(a5) # 1c000fec <__clz_tab+0x33c>
1c00bbee:	00000593          	li	a1,0
1c00bbf2:	1c001537          	lui	a0,0x1c001
1c00bbf6:	08f477d3          	fsub.s	a5,s0,a5
1c00bbfa:	1c001637          	lui	a2,0x1c001
1c00bbfe:	fe852703          	lw	a4,-24(a0) # 1c000fe8 <__clz_tab+0x338>
1c00bc02:	08862603          	lw	a2,136(a2) # 1c001088 <__clz_tab+0x3d8>
1c00bc06:	a0f595d3          	flt.s	a1,a1,a5
1c00bc0a:	10c77653          	fmul.s	a2,a4,a2
1c00bc0e:	30058063          	beqz	a1,1c00bf0e <cordic_cos_sin_fp32+0x400>
1c00bc12:	fe852583          	lw	a1,-24(a0)
1c00bc16:	08c5f653          	fsub.s	a2,a1,a2
1c00bc1a:	1c0015b7          	lui	a1,0x1c001
1c00bc1e:	ff05a583          	lw	a1,-16(a1) # 1c000ff0 <__clz_tab+0x340>
1c00bc22:	00b77753          	fadd.s	a4,a4,a1
1c00bc26:	1c0015b7          	lui	a1,0x1c001
1c00bc2a:	ff45a403          	lw	s0,-12(a1) # 1c000ff4 <__clz_tab+0x344>
1c00bc2e:	0887f7d3          	fsub.s	a5,a5,s0
1c00bc32:	00000513          	li	a0,0
1c00bc36:	1c0015b7          	lui	a1,0x1c001
1c00bc3a:	ff85a583          	lw	a1,-8(a1) # 1c000ff8 <__clz_tab+0x348>
1c00bc3e:	a0f51553          	flt.s	a0,a0,a5
1c00bc42:	10b77853          	fmul.s	a6,a4,a1
1c00bc46:	10b675d3          	fmul.s	a1,a2,a1
1c00bc4a:	3c050263          	beqz	a0,1c00c00e <cordic_cos_sin_fp32+0x500>
1c00bc4e:	00b77753          	fadd.s	a4,a4,a1
1c00bc52:	1c0015b7          	lui	a1,0x1c001
1c00bc56:	ffc5a583          	lw	a1,-4(a1) # 1c000ffc <__clz_tab+0x34c>
1c00bc5a:	09067653          	fsub.s	a2,a2,a6
1c00bc5e:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bc62:	00000513          	li	a0,0
1c00bc66:	1c0015b7          	lui	a1,0x1c001
1c00bc6a:	0005a583          	lw	a1,0(a1) # 1c001000 <__clz_tab+0x350>
1c00bc6e:	a0f51553          	flt.s	a0,a0,a5
1c00bc72:	10b77853          	fmul.s	a6,a4,a1
1c00bc76:	10b675d3          	fmul.s	a1,a2,a1
1c00bc7a:	36050f63          	beqz	a0,1c00bff8 <cordic_cos_sin_fp32+0x4ea>
1c00bc7e:	00b77753          	fadd.s	a4,a4,a1
1c00bc82:	1c0015b7          	lui	a1,0x1c001
1c00bc86:	0045a583          	lw	a1,4(a1) # 1c001004 <__clz_tab+0x354>
1c00bc8a:	09067653          	fsub.s	a2,a2,a6
1c00bc8e:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bc92:	00000513          	li	a0,0
1c00bc96:	1c0015b7          	lui	a1,0x1c001
1c00bc9a:	0085a583          	lw	a1,8(a1) # 1c001008 <__clz_tab+0x358>
1c00bc9e:	a0f51553          	flt.s	a0,a0,a5
1c00bca2:	10b77853          	fmul.s	a6,a4,a1
1c00bca6:	10b675d3          	fmul.s	a1,a2,a1
1c00bcaa:	32050c63          	beqz	a0,1c00bfe2 <cordic_cos_sin_fp32+0x4d4>
1c00bcae:	00b77753          	fadd.s	a4,a4,a1
1c00bcb2:	1c0015b7          	lui	a1,0x1c001
1c00bcb6:	00c5a583          	lw	a1,12(a1) # 1c00100c <__clz_tab+0x35c>
1c00bcba:	09067653          	fsub.s	a2,a2,a6
1c00bcbe:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bcc2:	00000513          	li	a0,0
1c00bcc6:	1c0015b7          	lui	a1,0x1c001
1c00bcca:	0105a583          	lw	a1,16(a1) # 1c001010 <__clz_tab+0x360>
1c00bcce:	a0f51553          	flt.s	a0,a0,a5
1c00bcd2:	10b77853          	fmul.s	a6,a4,a1
1c00bcd6:	10b675d3          	fmul.s	a1,a2,a1
1c00bcda:	2e050963          	beqz	a0,1c00bfcc <cordic_cos_sin_fp32+0x4be>
1c00bcde:	00b77753          	fadd.s	a4,a4,a1
1c00bce2:	1c0015b7          	lui	a1,0x1c001
1c00bce6:	0145a583          	lw	a1,20(a1) # 1c001014 <__clz_tab+0x364>
1c00bcea:	09067653          	fsub.s	a2,a2,a6
1c00bcee:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bcf2:	00000513          	li	a0,0
1c00bcf6:	1c0015b7          	lui	a1,0x1c001
1c00bcfa:	0185a583          	lw	a1,24(a1) # 1c001018 <__clz_tab+0x368>
1c00bcfe:	a0f51553          	flt.s	a0,a0,a5
1c00bd02:	10b77853          	fmul.s	a6,a4,a1
1c00bd06:	10b675d3          	fmul.s	a1,a2,a1
1c00bd0a:	2a050663          	beqz	a0,1c00bfb6 <cordic_cos_sin_fp32+0x4a8>
1c00bd0e:	00b77753          	fadd.s	a4,a4,a1
1c00bd12:	1c0015b7          	lui	a1,0x1c001
1c00bd16:	01c5a583          	lw	a1,28(a1) # 1c00101c <__clz_tab+0x36c>
1c00bd1a:	09067653          	fsub.s	a2,a2,a6
1c00bd1e:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bd22:	00000513          	li	a0,0
1c00bd26:	1c0015b7          	lui	a1,0x1c001
1c00bd2a:	0205a583          	lw	a1,32(a1) # 1c001020 <__clz_tab+0x370>
1c00bd2e:	a0f51553          	flt.s	a0,a0,a5
1c00bd32:	10b77853          	fmul.s	a6,a4,a1
1c00bd36:	10b675d3          	fmul.s	a1,a2,a1
1c00bd3a:	26050363          	beqz	a0,1c00bfa0 <cordic_cos_sin_fp32+0x492>
1c00bd3e:	00b77753          	fadd.s	a4,a4,a1
1c00bd42:	1c0015b7          	lui	a1,0x1c001
1c00bd46:	0245a583          	lw	a1,36(a1) # 1c001024 <__clz_tab+0x374>
1c00bd4a:	09067653          	fsub.s	a2,a2,a6
1c00bd4e:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bd52:	00000513          	li	a0,0
1c00bd56:	1c0015b7          	lui	a1,0x1c001
1c00bd5a:	0285a583          	lw	a1,40(a1) # 1c001028 <__clz_tab+0x378>
1c00bd5e:	a0f51553          	flt.s	a0,a0,a5
1c00bd62:	10b77853          	fmul.s	a6,a4,a1
1c00bd66:	10b675d3          	fmul.s	a1,a2,a1
1c00bd6a:	22050063          	beqz	a0,1c00bf8a <cordic_cos_sin_fp32+0x47c>
1c00bd6e:	00b77753          	fadd.s	a4,a4,a1
1c00bd72:	1c0015b7          	lui	a1,0x1c001
1c00bd76:	02c5a583          	lw	a1,44(a1) # 1c00102c <__clz_tab+0x37c>
1c00bd7a:	09067653          	fsub.s	a2,a2,a6
1c00bd7e:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bd82:	00000513          	li	a0,0
1c00bd86:	1c0015b7          	lui	a1,0x1c001
1c00bd8a:	0305a583          	lw	a1,48(a1) # 1c001030 <__clz_tab+0x380>
1c00bd8e:	a0f51553          	flt.s	a0,a0,a5
1c00bd92:	10b77853          	fmul.s	a6,a4,a1
1c00bd96:	10b675d3          	fmul.s	a1,a2,a1
1c00bd9a:	1c050d63          	beqz	a0,1c00bf74 <cordic_cos_sin_fp32+0x466>
1c00bd9e:	00b77753          	fadd.s	a4,a4,a1
1c00bda2:	1c0015b7          	lui	a1,0x1c001
1c00bda6:	0345a583          	lw	a1,52(a1) # 1c001034 <__clz_tab+0x384>
1c00bdaa:	09067653          	fsub.s	a2,a2,a6
1c00bdae:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bdb2:	00000513          	li	a0,0
1c00bdb6:	1c0015b7          	lui	a1,0x1c001
1c00bdba:	0385a583          	lw	a1,56(a1) # 1c001038 <__clz_tab+0x388>
1c00bdbe:	a0f51553          	flt.s	a0,a0,a5
1c00bdc2:	10b77853          	fmul.s	a6,a4,a1
1c00bdc6:	10b675d3          	fmul.s	a1,a2,a1
1c00bdca:	18050a63          	beqz	a0,1c00bf5e <cordic_cos_sin_fp32+0x450>
1c00bdce:	00b77753          	fadd.s	a4,a4,a1
1c00bdd2:	1c0015b7          	lui	a1,0x1c001
1c00bdd6:	03c5a583          	lw	a1,60(a1) # 1c00103c <__clz_tab+0x38c>
1c00bdda:	09067653          	fsub.s	a2,a2,a6
1c00bdde:	08b7f7d3          	fsub.s	a5,a5,a1
1c00bde2:	00000513          	li	a0,0
1c00bde6:	1c0015b7          	lui	a1,0x1c001
1c00bdea:	0405a583          	lw	a1,64(a1) # 1c001040 <__clz_tab+0x390>
1c00bdee:	a0f51553          	flt.s	a0,a0,a5
1c00bdf2:	10b77853          	fmul.s	a6,a4,a1
1c00bdf6:	10b675d3          	fmul.s	a1,a2,a1
1c00bdfa:	14050763          	beqz	a0,1c00bf48 <cordic_cos_sin_fp32+0x43a>
1c00bdfe:	00b77753          	fadd.s	a4,a4,a1
1c00be02:	1c0015b7          	lui	a1,0x1c001
1c00be06:	0445a583          	lw	a1,68(a1) # 1c001044 <__clz_tab+0x394>
1c00be0a:	09067653          	fsub.s	a2,a2,a6
1c00be0e:	08b7f7d3          	fsub.s	a5,a5,a1
1c00be12:	00000513          	li	a0,0
1c00be16:	1c0015b7          	lui	a1,0x1c001
1c00be1a:	0485a583          	lw	a1,72(a1) # 1c001048 <__clz_tab+0x398>
1c00be1e:	a0f51553          	flt.s	a0,a0,a5
1c00be22:	10b778d3          	fmul.s	a7,a4,a1
1c00be26:	10b67853          	fmul.s	a6,a2,a1
1c00be2a:	10050863          	beqz	a0,1c00bf3a <cordic_cos_sin_fp32+0x42c>
1c00be2e:	09167653          	fsub.s	a2,a2,a7
1c00be32:	01077753          	fadd.s	a4,a4,a6
1c00be36:	08b7f7d3          	fsub.s	a5,a5,a1
1c00be3a:	1c0015b7          	lui	a1,0x1c001
1c00be3e:	04c5a503          	lw	a0,76(a1) # 1c00104c <__clz_tab+0x39c>
1c00be42:	00000593          	li	a1,0
1c00be46:	a0f597d3          	flt.s	a5,a1,a5
1c00be4a:	10a775d3          	fmul.s	a1,a4,a0
1c00be4e:	10a67553          	fmul.s	a0,a2,a0
1c00be52:	cff9                	beqz	a5,1c00bf30 <cordic_cos_sin_fp32+0x422>
1c00be54:	08b67653          	fsub.s	a2,a2,a1
1c00be58:	00e57753          	fadd.s	a4,a0,a4
1c00be5c:	10d676d3          	fmul.s	a3,a2,a3
1c00be60:	40f2                	lw	ra,28(sp)
1c00be62:	4462                	lw	s0,24(sp)
1c00be64:	49b2                	lw	s3,12(sp)
1c00be66:	4a22                	lw	s4,8(sp)
1c00be68:	00d92023          	sw	a3,0(s2)
1c00be6c:	c098                	sw	a4,0(s1)
1c00be6e:	4942                	lw	s2,16(sp)
1c00be70:	44d2                	lw	s1,20(sp)
1c00be72:	4a92                	lw	s5,4(sp)
1c00be74:	4b02                	lw	s6,0(sp)
1c00be76:	6105                	addi	sp,sp,32
1c00be78:	8082                	ret
1c00be7a:	1c0017b7          	lui	a5,0x1c001
1c00be7e:	06078793          	addi	a5,a5,96 # 1c001060 <__clz_tab+0x3b0>
1c00be82:	4390                	lw	a2,0(a5)
1c00be84:	43d4                	lw	a3,4(a5)
1c00be86:	8552                	mv	a0,s4
1c00be88:	85d6                	mv	a1,s5
1c00be8a:	a9afe0ef          	jal	ra,1c00a124 <__ledf2>
1c00be8e:	d0055be3          	bgez	a0,1c00bba4 <cordic_cos_sin_fp32+0x96>
1c00be92:	0009a603          	lw	a2,0(s3)
1c00be96:	0049a683          	lw	a3,4(s3)
1c00be9a:	8552                	mv	a0,s4
1c00be9c:	85d6                	mv	a1,s5
1c00be9e:	b90fd0ef          	jal	ra,1c00922e <__adddf3>
1c00bea2:	b9d5                	j	1c00bb96 <cordic_cos_sin_fp32+0x88>
1c00bea4:	1c0017b7          	lui	a5,0x1c001
1c00bea8:	07078793          	addi	a5,a5,112 # 1c001070 <__clz_tab+0x3c0>
1c00beac:	4390                	lw	a2,0(a5)
1c00beae:	43d4                	lw	a3,4(a5)
1c00beb0:	8552                	mv	a0,s4
1c00beb2:	85d6                	mv	a1,s5
1c00beb4:	a70fe0ef          	jal	ra,1c00a124 <__ledf2>
1c00beb8:	00055963          	bgez	a0,1c00beca <cordic_cos_sin_fp32+0x3bc>
1c00bebc:	1c0017b7          	lui	a5,0x1c001
1c00bec0:	06078793          	addi	a5,a5,96 # 1c001060 <__clz_tab+0x3b0>
1c00bec4:	4388                	lw	a0,0(a5)
1c00bec6:	43cc                	lw	a1,4(a5)
1c00bec8:	b9f5                	j	1c00bbc4 <cordic_cos_sin_fp32+0xb6>
1c00beca:	1c0017b7          	lui	a5,0x1c001
1c00bece:	0907a683          	lw	a3,144(a5) # 1c001090 <__clz_tab+0x3e0>
1c00bed2:	00000793          	li	a5,0
1c00bed6:	a08797d3          	flt.s	a5,a5,s0
1c00beda:	d00796e3          	bnez	a5,1c00bbe6 <cordic_cos_sin_fp32+0xd8>
1c00bede:	1c0017b7          	lui	a5,0x1c001
1c00bee2:	fe47a703          	lw	a4,-28(a5) # 1c000fe4 <__clz_tab+0x334>
1c00bee6:	1c0017b7          	lui	a5,0x1c001
1c00beea:	fec7a783          	lw	a5,-20(a5) # 1c000fec <__clz_tab+0x33c>
1c00beee:	00000593          	li	a1,0
1c00bef2:	1c001637          	lui	a2,0x1c001
1c00bef6:	00f477d3          	fadd.s	a5,s0,a5
1c00befa:	08862603          	lw	a2,136(a2) # 1c001088 <__clz_tab+0x3d8>
1c00befe:	1c001537          	lui	a0,0x1c001
1c00bf02:	10c77653          	fmul.s	a2,a4,a2
1c00bf06:	a0f595d3          	flt.s	a1,a1,a5
1c00bf0a:	d00594e3          	bnez	a1,1c00bc12 <cordic_cos_sin_fp32+0x104>
1c00bf0e:	fe852583          	lw	a1,-24(a0) # 1c000fe8 <__clz_tab+0x338>
1c00bf12:	00b67653          	fadd.s	a2,a2,a1
1c00bf16:	1c0015b7          	lui	a1,0x1c001
1c00bf1a:	ff05a583          	lw	a1,-16(a1) # 1c000ff0 <__clz_tab+0x340>
1c00bf1e:	08b77753          	fsub.s	a4,a4,a1
1c00bf22:	1c0015b7          	lui	a1,0x1c001
1c00bf26:	ff45a403          	lw	s0,-12(a1) # 1c000ff4 <__clz_tab+0x344>
1c00bf2a:	0087f7d3          	fadd.s	a5,a5,s0
1c00bf2e:	b311                	j	1c00bc32 <cordic_cos_sin_fp32+0x124>
1c00bf30:	00c5f653          	fadd.s	a2,a1,a2
1c00bf34:	08a77753          	fsub.s	a4,a4,a0
1c00bf38:	b715                	j	1c00be5c <cordic_cos_sin_fp32+0x34e>
1c00bf3a:	01167653          	fadd.s	a2,a2,a7
1c00bf3e:	09077753          	fsub.s	a4,a4,a6
1c00bf42:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bf46:	bdd5                	j	1c00be3a <cordic_cos_sin_fp32+0x32c>
1c00bf48:	08b77753          	fsub.s	a4,a4,a1
1c00bf4c:	1c0015b7          	lui	a1,0x1c001
1c00bf50:	0445a583          	lw	a1,68(a1) # 1c001044 <__clz_tab+0x394>
1c00bf54:	01067653          	fadd.s	a2,a2,a6
1c00bf58:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bf5c:	bd5d                	j	1c00be12 <cordic_cos_sin_fp32+0x304>
1c00bf5e:	08b77753          	fsub.s	a4,a4,a1
1c00bf62:	1c0015b7          	lui	a1,0x1c001
1c00bf66:	03c5a583          	lw	a1,60(a1) # 1c00103c <__clz_tab+0x38c>
1c00bf6a:	01067653          	fadd.s	a2,a2,a6
1c00bf6e:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bf72:	bd85                	j	1c00bde2 <cordic_cos_sin_fp32+0x2d4>
1c00bf74:	08b77753          	fsub.s	a4,a4,a1
1c00bf78:	1c0015b7          	lui	a1,0x1c001
1c00bf7c:	0345a583          	lw	a1,52(a1) # 1c001034 <__clz_tab+0x384>
1c00bf80:	01067653          	fadd.s	a2,a2,a6
1c00bf84:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bf88:	b52d                	j	1c00bdb2 <cordic_cos_sin_fp32+0x2a4>
1c00bf8a:	08b77753          	fsub.s	a4,a4,a1
1c00bf8e:	1c0015b7          	lui	a1,0x1c001
1c00bf92:	02c5a583          	lw	a1,44(a1) # 1c00102c <__clz_tab+0x37c>
1c00bf96:	01067653          	fadd.s	a2,a2,a6
1c00bf9a:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bf9e:	b3d5                	j	1c00bd82 <cordic_cos_sin_fp32+0x274>
1c00bfa0:	08b77753          	fsub.s	a4,a4,a1
1c00bfa4:	1c0015b7          	lui	a1,0x1c001
1c00bfa8:	0245a583          	lw	a1,36(a1) # 1c001024 <__clz_tab+0x374>
1c00bfac:	01067653          	fadd.s	a2,a2,a6
1c00bfb0:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bfb4:	bb79                	j	1c00bd52 <cordic_cos_sin_fp32+0x244>
1c00bfb6:	08b77753          	fsub.s	a4,a4,a1
1c00bfba:	1c0015b7          	lui	a1,0x1c001
1c00bfbe:	01c5a583          	lw	a1,28(a1) # 1c00101c <__clz_tab+0x36c>
1c00bfc2:	01067653          	fadd.s	a2,a2,a6
1c00bfc6:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bfca:	bba1                	j	1c00bd22 <cordic_cos_sin_fp32+0x214>
1c00bfcc:	08b77753          	fsub.s	a4,a4,a1
1c00bfd0:	1c0015b7          	lui	a1,0x1c001
1c00bfd4:	0145a583          	lw	a1,20(a1) # 1c001014 <__clz_tab+0x364>
1c00bfd8:	01067653          	fadd.s	a2,a2,a6
1c00bfdc:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bfe0:	bb09                	j	1c00bcf2 <cordic_cos_sin_fp32+0x1e4>
1c00bfe2:	08b77753          	fsub.s	a4,a4,a1
1c00bfe6:	1c0015b7          	lui	a1,0x1c001
1c00bfea:	00c5a583          	lw	a1,12(a1) # 1c00100c <__clz_tab+0x35c>
1c00bfee:	01067653          	fadd.s	a2,a2,a6
1c00bff2:	00b7f7d3          	fadd.s	a5,a5,a1
1c00bff6:	b1f1                	j	1c00bcc2 <cordic_cos_sin_fp32+0x1b4>
1c00bff8:	08b77753          	fsub.s	a4,a4,a1
1c00bffc:	1c0015b7          	lui	a1,0x1c001
1c00c000:	0045a583          	lw	a1,4(a1) # 1c001004 <__clz_tab+0x354>
1c00c004:	01067653          	fadd.s	a2,a2,a6
1c00c008:	00b7f7d3          	fadd.s	a5,a5,a1
1c00c00c:	b159                	j	1c00bc92 <cordic_cos_sin_fp32+0x184>
1c00c00e:	08b77753          	fsub.s	a4,a4,a1
1c00c012:	1c0015b7          	lui	a1,0x1c001
1c00c016:	ffc5a583          	lw	a1,-4(a1) # 1c000ffc <__clz_tab+0x34c>
1c00c01a:	01067653          	fadd.s	a2,a2,a6
1c00c01e:	00b7f7d3          	fadd.s	a5,a5,a1
1c00c022:	b181                	j	1c00bc62 <cordic_cos_sin_fp32+0x154>

1c00c024 <pos_fll_set_freq>:
1c00c024:	100596b3          	p.fl1	a3,a1
1c00c028:	47f5                	li	a5,29
1c00c02a:	4705                	li	a4,1
1c00c02c:	82d7b7db          	p.subun	a5,a5,a3,1
1c00c030:	04f767b3          	p.max	a5,a4,a5
1c00c034:	00400693          	li	a3,4
1c00c038:	fff78813          	addi	a6,a5,-1
1c00c03c:	00f595b3          	sll	a1,a1,a5
1c00c040:	00a68633          	add	a2,a3,a0
1c00c044:	01071733          	sll	a4,a4,a6
1c00c048:	00c64603          	lbu	a2,12(a2)
1c00c04c:	dc05b833          	p.bclr	a6,a1,14,0
1c00c050:	c0f8275b          	p.addunr	a4,a6,a5
1c00c054:	00251813          	slli	a6,a0,0x2
1c00c058:	00e6e823          	p.sw	a4,a6(a3)
1c00c05c:	ce19                	beqz	a2,1c00c07a <pos_fll_set_freq+0x56>
1c00c05e:	0512                	slli	a0,a0,0x4
1c00c060:	0511                	addi	a0,a0,4
1c00c062:	1a100637          	lui	a2,0x1a100
1c00c066:	20a67683          	p.lw	a3,a0(a2)
1c00c06a:	81bd                	srli	a1,a1,0xf
1c00c06c:	de05a6b3          	p.insert	a3,a1,15,0
1c00c070:	0785                	addi	a5,a5,1
1c00c072:	c7a7a6b3          	p.insert	a3,a5,3,26
1c00c076:	00d66523          	p.sw	a3,a0(a2)
1c00c07a:	853a                	mv	a0,a4
1c00c07c:	8082                	ret

1c00c07e <pos_fll_init>:
1c00c07e:	00451593          	slli	a1,a0,0x4
1c00c082:	832a                	mv	t1,a0
1c00c084:	00458e13          	addi	t3,a1,4
1c00c088:	1a100637          	lui	a2,0x1a100
1c00c08c:	21c67703          	p.lw	a4,t3(a2)
1c00c090:	86ba                	mv	a3,a4
1c00c092:	04074163          	bltz	a4,1c00c0d4 <pos_fll_init+0x56>
1c00c096:	00858793          	addi	a5,a1,8
1c00c09a:	20f67883          	p.lw	a7,a5(a2)
1c00c09e:	4519                	li	a0,6
1c00c0a0:	caa528b3          	p.insert	a7,a0,5,10
1c00c0a4:	05000513          	li	a0,80
1c00c0a8:	d70528b3          	p.insert	a7,a0,11,16
1c00c0ac:	1a100837          	lui	a6,0x1a100
1c00c0b0:	011867a3          	p.sw	a7,a5(a6)
1c00c0b4:	05b1                	addi	a1,a1,12
1c00c0b6:	20b67603          	p.lw	a2,a1(a2)
1c00c0ba:	14c00793          	li	a5,332
1c00c0be:	d307a633          	p.insert	a2,a5,9,16
1c00c0c2:	00c865a3          	p.sw	a2,a1(a6)
1c00c0c6:	4785                	li	a5,1
1c00c0c8:	c1e7a6b3          	p.insert	a3,a5,0,30
1c00c0cc:	c1f7a6b3          	p.insert	a3,a5,0,31
1c00c0d0:	00d86e23          	p.sw	a3,t3(a6)
1c00c0d4:	00400793          	li	a5,4
1c00c0d8:	00231613          	slli	a2,t1,0x2
1c00c0dc:	963e                	add	a2,a2,a5
1c00c0de:	4208                	lw	a0,0(a2)
1c00c0e0:	ed19                	bnez	a0,1c00c0fe <pos_fll_init+0x80>
1c00c0e2:	10075733          	p.exthz	a4,a4
1c00c0e6:	c7a696b3          	p.extractu	a3,a3,3,26
1c00c0ea:	073e                	slli	a4,a4,0xf
1c00c0ec:	16fd                	addi	a3,a3,-1
1c00c0ee:	00d75533          	srl	a0,a4,a3
1c00c0f2:	c208                	sw	a0,0(a2)
1c00c0f4:	979a                	add	a5,a5,t1
1c00c0f6:	4705                	li	a4,1
1c00c0f8:	00e78623          	sb	a4,12(a5)
1c00c0fc:	8082                	ret
1c00c0fe:	46f5                	li	a3,29
1c00c100:	10051733          	p.fl1	a4,a0
1c00c104:	4585                	li	a1,1
1c00c106:	82e6b75b          	p.subun	a4,a3,a4,1
1c00c10a:	04e5e733          	p.max	a4,a1,a4
1c00c10e:	fff70893          	addi	a7,a4,-1
1c00c112:	00e51833          	sll	a6,a0,a4
1c00c116:	006786b3          	add	a3,a5,t1
1c00c11a:	011595b3          	sll	a1,a1,a7
1c00c11e:	00c6c683          	lbu	a3,12(a3)
1c00c122:	dc0838b3          	p.bclr	a7,a6,14,0
1c00c126:	c0e8a5db          	p.addunr	a1,a7,a4
1c00c12a:	c20c                	sw	a1,0(a2)
1c00c12c:	d6e1                	beqz	a3,1c00c0f4 <pos_fll_init+0x76>
1c00c12e:	1a100637          	lui	a2,0x1a100
1c00c132:	21c67683          	p.lw	a3,t3(a2)
1c00c136:	00f85813          	srli	a6,a6,0xf
1c00c13a:	de0826b3          	p.insert	a3,a6,15,0
1c00c13e:	0705                	addi	a4,a4,1
1c00c140:	c7a726b3          	p.insert	a3,a4,3,26
1c00c144:	00d66e23          	p.sw	a3,t3(a2)
1c00c148:	979a                	add	a5,a5,t1
1c00c14a:	4705                	li	a4,1
1c00c14c:	00e78623          	sb	a4,12(a5)
1c00c150:	8082                	ret

1c00c152 <pos_fll_constructor>:
1c00c152:	00400793          	li	a5,4
1c00c156:	0007a023          	sw	zero,0(a5)
1c00c15a:	0007a223          	sw	zero,4(a5)
1c00c15e:	0007a423          	sw	zero,8(a5)
1c00c162:	00079623          	sh	zero,12(a5)
1c00c166:	00078723          	sb	zero,14(a5)
1c00c16a:	8082                	ret

1c00c16c <pos_soc_init>:
1c00c16c:	1141                	addi	sp,sp,-16
1c00c16e:	c606                	sw	ra,12(sp)
1c00c170:	c422                	sw	s0,8(sp)
1c00c172:	37c5                	jal	1c00c152 <pos_fll_constructor>
1c00c174:	4501                	li	a0,0
1c00c176:	3721                	jal	1c00c07e <pos_fll_init>
1c00c178:	00a02a23          	sw	a0,20(zero) # 14 <pos_freq_domains>
1c00c17c:	4505                	li	a0,1
1c00c17e:	3701                	jal	1c00c07e <pos_fll_init>
1c00c180:	01400413          	li	s0,20
1c00c184:	c408                	sw	a0,8(s0)
1c00c186:	4509                	li	a0,2
1c00c188:	3ddd                	jal	1c00c07e <pos_fll_init>
1c00c18a:	c048                	sw	a0,4(s0)
1c00c18c:	40b2                	lw	ra,12(sp)
1c00c18e:	4422                	lw	s0,8(sp)
1c00c190:	0141                	addi	sp,sp,16
1c00c192:	8082                	ret

1c00c194 <strchr>:
1c00c194:	00054783          	lbu	a5,0(a0)
1c00c198:	0ff5f593          	andi	a1,a1,255
1c00c19c:	00b78c63          	beq	a5,a1,1c00c1b4 <strchr+0x20>
1c00c1a0:	cb99                	beqz	a5,1c00c1b6 <strchr+0x22>
1c00c1a2:	00150793          	addi	a5,a0,1
1c00c1a6:	a011                	j	1c00c1aa <strchr+0x16>
1c00c1a8:	c719                	beqz	a4,1c00c1b6 <strchr+0x22>
1c00c1aa:	853e                	mv	a0,a5
1c00c1ac:	0017c70b          	p.lbu	a4,1(a5!)
1c00c1b0:	feb71ce3          	bne	a4,a1,1c00c1a8 <strchr+0x14>
1c00c1b4:	8082                	ret
1c00c1b6:	0015b593          	seqz	a1,a1
1c00c1ba:	40b005b3          	neg	a1,a1
1c00c1be:	8d6d                	and	a0,a0,a1
1c00c1c0:	8082                	ret

1c00c1c2 <puts>:
1c00c1c2:	00054783          	lbu	a5,0(a0)
1c00c1c6:	f14026f3          	csrr	a3,mhartid
1c00c1ca:	c395                	beqz	a5,1c00c1ee <puts+0x2c>
1c00c1cc:	ca569733          	p.extractu	a4,a3,5,5
1c00c1d0:	f1402673          	csrr	a2,mhartid
1c00c1d4:	071e                	slli	a4,a4,0x7
1c00c1d6:	060e                	slli	a2,a2,0x3
1c00c1d8:	1a10f5b7          	lui	a1,0x1a10f
1c00c1dc:	972e                	add	a4,a4,a1
1c00c1de:	ee863633          	p.bclr	a2,a2,23,8
1c00c1e2:	0505                	addi	a0,a0,1
1c00c1e4:	9732                	add	a4,a4,a2
1c00c1e6:	c31c                	sw	a5,0(a4)
1c00c1e8:	0015478b          	p.lbu	a5,1(a0!)
1c00c1ec:	ffed                	bnez	a5,1c00c1e6 <puts+0x24>
1c00c1ee:	ca5696b3          	p.extractu	a3,a3,5,5
1c00c1f2:	1a10f737          	lui	a4,0x1a10f
1c00c1f6:	f14027f3          	csrr	a5,mhartid
1c00c1fa:	069e                	slli	a3,a3,0x7
1c00c1fc:	078e                	slli	a5,a5,0x3
1c00c1fe:	96ba                	add	a3,a3,a4
1c00c200:	ee87b7b3          	p.bclr	a5,a5,23,8
1c00c204:	4729                	li	a4,10
1c00c206:	00e6e7a3          	p.sw	a4,a5(a3)
1c00c20a:	4501                	li	a0,0
1c00c20c:	8082                	ret

1c00c20e <pos_libc_fputc_locked>:
1c00c20e:	f14027f3          	csrr	a5,mhartid
1c00c212:	1a10f6b7          	lui	a3,0x1a10f
1c00c216:	f1402773          	csrr	a4,mhartid
1c00c21a:	ca5797b3          	p.extractu	a5,a5,5,5
1c00c21e:	070e                	slli	a4,a4,0x3
1c00c220:	079e                	slli	a5,a5,0x7
1c00c222:	0ff57513          	andi	a0,a0,255
1c00c226:	ee873733          	p.bclr	a4,a4,23,8
1c00c22a:	97b6                	add	a5,a5,a3
1c00c22c:	00a767a3          	p.sw	a0,a5(a4)
1c00c230:	4501                	li	a0,0
1c00c232:	8082                	ret

1c00c234 <putchar>:
1c00c234:	f14027f3          	csrr	a5,mhartid
1c00c238:	1a10f6b7          	lui	a3,0x1a10f
1c00c23c:	f1402773          	csrr	a4,mhartid
1c00c240:	ca5797b3          	p.extractu	a5,a5,5,5
1c00c244:	070e                	slli	a4,a4,0x3
1c00c246:	079e                	slli	a5,a5,0x7
1c00c248:	0ff57513          	andi	a0,a0,255
1c00c24c:	ee873733          	p.bclr	a4,a4,23,8
1c00c250:	97b6                	add	a5,a5,a3
1c00c252:	00a767a3          	p.sw	a0,a5(a4)
1c00c256:	4501                	li	a0,0
1c00c258:	8082                	ret

1c00c25a <pos_libc_prf_locked>:
1c00c25a:	a095                	j	1c00c2be <pos_libc_prf>

1c00c25c <exit>:
1c00c25c:	1141                	addi	sp,sp,-16
1c00c25e:	c422                	sw	s0,8(sp)
1c00c260:	c606                	sw	ra,12(sp)
1c00c262:	842a                	mv	s0,a0
1c00c264:	443010ef          	jal	ra,1c00dea6 <pos_init_stop>
1c00c268:	c1f44533          	p.bset	a0,s0,0,31
1c00c26c:	1a1047b7          	lui	a5,0x1a104
1c00c270:	0aa7a023          	sw	a0,160(a5) # 1a1040a0 <__l1_end+0xa10371c>
1c00c274:	1a10a7b7          	lui	a5,0x1a10a
1c00c278:	577d                	li	a4,-1
1c00c27a:	80e7a423          	sw	a4,-2040(a5) # 1a109808 <__l1_end+0xa108e84>
1c00c27e:	10500073          	wfi
1c00c282:	10500073          	wfi
1c00c286:	bfe5                	j	1c00c27e <exit+0x22>

1c00c288 <pos_io_start>:
1c00c288:	4501                	li	a0,0
1c00c28a:	8082                	ret

1c00c28c <pos_io_stop>:
1c00c28c:	4501                	li	a0,0
1c00c28e:	8082                	ret

1c00c290 <printf>:
1c00c290:	7139                	addi	sp,sp,-64
1c00c292:	02410313          	addi	t1,sp,36
1c00c296:	d432                	sw	a2,40(sp)
1c00c298:	862a                	mv	a2,a0
1c00c29a:	1c00c537          	lui	a0,0x1c00c
1c00c29e:	d22e                	sw	a1,36(sp)
1c00c2a0:	d636                	sw	a3,44(sp)
1c00c2a2:	4589                	li	a1,2
1c00c2a4:	869a                	mv	a3,t1
1c00c2a6:	20e50513          	addi	a0,a0,526 # 1c00c20e <pos_libc_fputc_locked>
1c00c2aa:	ce06                	sw	ra,28(sp)
1c00c2ac:	d83a                	sw	a4,48(sp)
1c00c2ae:	da3e                	sw	a5,52(sp)
1c00c2b0:	dc42                	sw	a6,56(sp)
1c00c2b2:	de46                	sw	a7,60(sp)
1c00c2b4:	c61a                	sw	t1,12(sp)
1c00c2b6:	3755                	jal	1c00c25a <pos_libc_prf_locked>
1c00c2b8:	40f2                	lw	ra,28(sp)
1c00c2ba:	6121                	addi	sp,sp,64
1c00c2bc:	8082                	ret

1c00c2be <pos_libc_prf>:
1c00c2be:	7119                	addi	sp,sp,-128
1c00c2c0:	dca2                	sw	s0,120(sp)
1c00c2c2:	1058                	addi	a4,sp,36
1c00c2c4:	4401                	li	s0,0
1c00c2c6:	d6ce                	sw	s3,108(sp)
1c00c2c8:	d4d2                	sw	s4,104(sp)
1c00c2ca:	d2d6                	sw	s5,100(sp)
1c00c2cc:	cede                	sw	s7,92(sp)
1c00c2ce:	cce2                	sw	s8,88(sp)
1c00c2d0:	c8ea                	sw	s10,80(sp)
1c00c2d2:	de86                	sw	ra,124(sp)
1c00c2d4:	daa6                	sw	s1,116(sp)
1c00c2d6:	d8ca                	sw	s2,112(sp)
1c00c2d8:	d0da                	sw	s6,96(sp)
1c00c2da:	cae6                	sw	s9,84(sp)
1c00c2dc:	c6ee                	sw	s11,76(sp)
1c00c2de:	89aa                	mv	s3,a0
1c00c2e0:	8bae                	mv	s7,a1
1c00c2e2:	8d32                	mv	s10,a2
1c00c2e4:	8a36                	mv	s4,a3
1c00c2e6:	1c001c37          	lui	s8,0x1c001
1c00c2ea:	c03a                	sw	a4,0(sp)
1c00c2ec:	8aa2                	mv	s5,s0
1c00c2ee:	a039                	j	1c00c2fc <pos_libc_prf+0x3e>
1c00c2f0:	85de                	mv	a1,s7
1c00c2f2:	9982                	jalr	s3
1c00c2f4:	15f52063          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c2f8:	0a85                	addi	s5,s5,1
1c00c2fa:	8d22                	mv	s10,s0
1c00c2fc:	000d4503          	lbu	a0,0(s10)
1c00c300:	001d0413          	addi	s0,s10,1
1c00c304:	c979                	beqz	a0,1c00c3da <pos_libc_prf+0x11c>
1c00c306:	02500793          	li	a5,37
1c00c30a:	fef513e3          	bne	a0,a5,1c00c2f0 <pos_libc_prf+0x32>
1c00c30e:	ca02                	sw	zero,20(sp)
1c00c310:	c802                	sw	zero,16(sp)
1c00c312:	c602                	sw	zero,12(sp)
1c00c314:	c402                	sw	zero,8(sp)
1c00c316:	c202                	sw	zero,4(sp)
1c00c318:	02300493          	li	s1,35
1c00c31c:	02d00b13          	li	s6,45
1c00c320:	00044c83          	lbu	s9,0(s0)
1c00c324:	00140913          	addi	s2,s0,1
1c00c328:	0a4c0513          	addi	a0,s8,164 # 1c0010a4 <__clz_tab+0x3f4>
1c00c32c:	85e6                	mv	a1,s9
1c00c32e:	8d4a                	mv	s10,s2
1c00c330:	8de6                	mv	s11,s9
1c00c332:	358d                	jal	1c00c194 <strchr>
1c00c334:	c91d                	beqz	a0,1c00c36a <pos_libc_prf+0xac>
1c00c336:	109c8963          	beq	s9,s1,1c00c448 <pos_libc_prf+0x18a>
1c00c33a:	0194fe63          	bleu	s9,s1,1c00c356 <pos_libc_prf+0x98>
1c00c33e:	116c8163          	beq	s9,s6,1c00c440 <pos_libc_prf+0x182>
1c00c342:	03000713          	li	a4,48
1c00c346:	0eec8963          	beq	s9,a4,1c00c438 <pos_libc_prf+0x17a>
1c00c34a:	02b00793          	li	a5,43
1c00c34e:	0afc8763          	beq	s9,a5,1c00c3fc <pos_libc_prf+0x13e>
1c00c352:	844a                	mv	s0,s2
1c00c354:	b7f1                	j	1c00c320 <pos_libc_prf+0x62>
1c00c356:	080c8263          	beqz	s9,1c00c3da <pos_libc_prf+0x11c>
1c00c35a:	02000793          	li	a5,32
1c00c35e:	fefc9ae3          	bne	s9,a5,1c00c352 <pos_libc_prf+0x94>
1c00c362:	4785                	li	a5,1
1c00c364:	c83e                	sw	a5,16(sp)
1c00c366:	844a                	mv	s0,s2
1c00c368:	bf65                	j	1c00c320 <pos_libc_prf+0x62>
1c00c36a:	02a00693          	li	a3,42
1c00c36e:	2edc8263          	beq	s9,a3,1c00c652 <pos_libc_prf+0x394>
1c00c372:	fd0c8693          	addi	a3,s9,-48
1c00c376:	45a5                	li	a1,9
1c00c378:	4b01                	li	s6,0
1c00c37a:	0cd5fd63          	bleu	a3,a1,1c00c454 <pos_libc_prf+0x196>
1c00c37e:	02e00793          	li	a5,46
1c00c382:	28fd8663          	beq	s11,a5,1c00c60e <pos_libc_prf+0x350>
1c00c386:	846e                	mv	s0,s11
1c00c388:	54fd                	li	s1,-1
1c00c38a:	1c001737          	lui	a4,0x1c001
1c00c38e:	85ee                	mv	a1,s11
1c00c390:	0ac70513          	addi	a0,a4,172 # 1c0010ac <__clz_tab+0x3fc>
1c00c394:	3501                	jal	1c00c194 <strchr>
1c00c396:	4781                	li	a5,0
1c00c398:	cd19                	beqz	a0,1c00c3b6 <pos_libc_prf+0xf8>
1c00c39a:	06c00793          	li	a5,108
1c00c39e:	001d0693          	addi	a3,s10,1
1c00c3a2:	000d4d83          	lbu	s11,0(s10)
1c00c3a6:	3af40563          	beq	s0,a5,1c00c750 <pos_libc_prf+0x492>
1c00c3aa:	06800793          	li	a5,104
1c00c3ae:	3af40563          	beq	s0,a5,1c00c758 <pos_libc_prf+0x49a>
1c00c3b2:	8d36                	mv	s10,a3
1c00c3b4:	87a2                	mv	a5,s0
1c00c3b6:	06700693          	li	a3,103
1c00c3ba:	0bb6ee63          	bltu	a3,s11,1c00c476 <pos_libc_prf+0x1b8>
1c00c3be:	06500693          	li	a3,101
1c00c3c2:	3addf0e3          	bleu	a3,s11,1c00cf62 <pos_libc_prf+0xca4>
1c00c3c6:	04700693          	li	a3,71
1c00c3ca:	39b6ed63          	bltu	a3,s11,1c00c764 <pos_libc_prf+0x4a6>
1c00c3ce:	04500793          	li	a5,69
1c00c3d2:	38fdf8e3          	bleu	a5,s11,1c00cf62 <pos_libc_prf+0xca4>
1c00c3d6:	2e0d9863          	bnez	s11,1c00c6c6 <pos_libc_prf+0x408>
1c00c3da:	8456                	mv	s0,s5
1c00c3dc:	8522                	mv	a0,s0
1c00c3de:	50f6                	lw	ra,124(sp)
1c00c3e0:	5466                	lw	s0,120(sp)
1c00c3e2:	54d6                	lw	s1,116(sp)
1c00c3e4:	5946                	lw	s2,112(sp)
1c00c3e6:	59b6                	lw	s3,108(sp)
1c00c3e8:	5a26                	lw	s4,104(sp)
1c00c3ea:	5a96                	lw	s5,100(sp)
1c00c3ec:	5b06                	lw	s6,96(sp)
1c00c3ee:	4bf6                	lw	s7,92(sp)
1c00c3f0:	4c66                	lw	s8,88(sp)
1c00c3f2:	4cd6                	lw	s9,84(sp)
1c00c3f4:	4d46                	lw	s10,80(sp)
1c00c3f6:	4db6                	lw	s11,76(sp)
1c00c3f8:	6109                	addi	sp,sp,128
1c00c3fa:	8082                	ret
1c00c3fc:	4705                	li	a4,1
1c00c3fe:	c63a                	sw	a4,12(sp)
1c00c400:	844a                	mv	s0,s2
1c00c402:	bf39                	j	1c00c320 <pos_libc_prf+0x62>
1c00c404:	00044503          	lbu	a0,0(s0)
1c00c408:	47a5                	li	a5,9
1c00c40a:	fd050693          	addi	a3,a0,-48
1c00c40e:	72d7ece3          	bltu	a5,a3,1c00d346 <pos_libc_prf+0x1088>
1c00c412:	00140793          	addi	a5,s0,1
1c00c416:	8dbe                	mv	s11,a5
1c00c418:	a809                	j	1c00c42a <pos_libc_prf+0x16c>
1c00c41a:	001dc50b          	p.lbu	a0,1(s11!)
1c00c41e:	4725                	li	a4,9
1c00c420:	197d                	addi	s2,s2,-1
1c00c422:	fd050793          	addi	a5,a0,-48
1c00c426:	72f761e3          	bltu	a4,a5,1c00d348 <pos_libc_prf+0x108a>
1c00c42a:	85de                	mv	a1,s7
1c00c42c:	846e                	mv	s0,s11
1c00c42e:	9982                	jalr	s3
1c00c430:	fff535e3          	p.bneimm	a0,-1,1c00c41a <pos_libc_prf+0x15c>
1c00c434:	547d                	li	s0,-1
1c00c436:	b75d                	j	1c00c3dc <pos_libc_prf+0x11e>
1c00c438:	4785                	li	a5,1
1c00c43a:	ca3e                	sw	a5,20(sp)
1c00c43c:	844a                	mv	s0,s2
1c00c43e:	b5cd                	j	1c00c320 <pos_libc_prf+0x62>
1c00c440:	4705                	li	a4,1
1c00c442:	c43a                	sw	a4,8(sp)
1c00c444:	844a                	mv	s0,s2
1c00c446:	bde9                	j	1c00c320 <pos_libc_prf+0x62>
1c00c448:	4705                	li	a4,1
1c00c44a:	c23a                	sw	a4,4(sp)
1c00c44c:	844a                	mv	s0,s2
1c00c44e:	bdc9                	j	1c00c320 <pos_libc_prf+0x62>
1c00c450:	844a                	mv	s0,s2
1c00c452:	0905                	addi	s2,s2,1
1c00c454:	00094d83          	lbu	s11,0(s2)
1c00c458:	002b1793          	slli	a5,s6,0x2
1c00c45c:	97da                	add	a5,a5,s6
1c00c45e:	0786                	slli	a5,a5,0x1
1c00c460:	97e6                	add	a5,a5,s9
1c00c462:	fd0d8713          	addi	a4,s11,-48
1c00c466:	fd078b13          	addi	s6,a5,-48
1c00c46a:	8cee                	mv	s9,s11
1c00c46c:	fee5f2e3          	bleu	a4,a1,1c00c450 <pos_libc_prf+0x192>
1c00c470:	00240d13          	addi	s10,s0,2
1c00c474:	b729                	j	1c00c37e <pos_libc_prf+0xc0>
1c00c476:	07000693          	li	a3,112
1c00c47a:	24dd83e3          	beq	s11,a3,1c00cec0 <pos_libc_prf+0xc02>
1c00c47e:	1fb6e463          	bltu	a3,s11,1c00c666 <pos_libc_prf+0x3a8>
1c00c482:	06e00693          	li	a3,110
1c00c486:	20dd88e3          	beq	s11,a3,1c00ce96 <pos_libc_prf+0xbd8>
1c00c48a:	2fb6e763          	bltu	a3,s11,1c00c778 <pos_libc_prf+0x4ba>
1c00c48e:	06900693          	li	a3,105
1c00c492:	2add9ae3          	bne	s11,a3,1c00cf46 <pos_libc_prf+0xc88>
1c00c496:	06c00693          	li	a3,108
1c00c49a:	00d78a63          	beq	a5,a3,1c00c4ae <pos_libc_prf+0x1f0>
1c00c49e:	07a00693          	li	a3,122
1c00c4a2:	00d78663          	beq	a5,a3,1c00c4ae <pos_libc_prf+0x1f0>
1c00c4a6:	04c00693          	li	a3,76
1c00c4aa:	60d780e3          	beq	a5,a3,1c00d2aa <pos_libc_prf+0xfec>
1c00c4ae:	000a2783          	lw	a5,0(s4)
1c00c4b2:	0a11                	addi	s4,s4,4
1c00c4b4:	ce52                	sw	s4,28(sp)
1c00c4b6:	41f7d713          	srai	a4,a5,0x1f
1c00c4ba:	c23a                	sw	a4,4(sp)
1c00c4bc:	4e0744e3          	bltz	a4,1c00d1a4 <pos_libc_prf+0xee6>
1c00c4c0:	4732                	lw	a4,12(sp)
1c00c4c2:	44070ce3          	beqz	a4,1c00d11a <pos_libc_prf+0xe5c>
1c00c4c6:	4c92                	lw	s9,4(sp)
1c00c4c8:	02b00693          	li	a3,43
1c00c4cc:	02d10223          	sb	a3,36(sp)
1c00c4d0:	02510a13          	addi	s4,sp,37
1c00c4d4:	843e                	mv	s0,a5
1c00c4d6:	8dd2                	mv	s11,s4
1c00c4d8:	a011                	j	1c00c4dc <pos_libc_prf+0x21e>
1c00c4da:	8dca                	mv	s11,s2
1c00c4dc:	4629                	li	a2,10
1c00c4de:	4681                	li	a3,0
1c00c4e0:	8522                	mv	a0,s0
1c00c4e2:	85e6                	mv	a1,s9
1c00c4e4:	a1dfc0ef          	jal	ra,1c008f00 <__umoddi3>
1c00c4e8:	892a                	mv	s2,a0
1c00c4ea:	85e6                	mv	a1,s9
1c00c4ec:	8522                	mv	a0,s0
1c00c4ee:	4629                	li	a2,10
1c00c4f0:	4681                	li	a3,0
1c00c4f2:	ed2fc0ef          	jal	ra,1c008bc4 <__udivdi3>
1c00c4f6:	03090913          	addi	s2,s2,48
1c00c4fa:	012d8023          	sb	s2,0(s11)
1c00c4fe:	00a5e7b3          	or	a5,a1,a0
1c00c502:	001d8913          	addi	s2,s11,1
1c00c506:	842a                	mv	s0,a0
1c00c508:	8cae                	mv	s9,a1
1c00c50a:	fbe1                	bnez	a5,1c00c4da <pos_libc_prf+0x21c>
1c00c50c:	00090023          	sb	zero,0(s2)
1c00c510:	01ba7d63          	bleu	s11,s4,1c00c52a <pos_libc_prf+0x26c>
1c00c514:	87ee                	mv	a5,s11
1c00c516:	000a4683          	lbu	a3,0(s4)
1c00c51a:	0007c703          	lbu	a4,0(a5)
1c00c51e:	fed78fab          	p.sb	a3,-1(a5!)
1c00c522:	00ea00ab          	p.sb	a4,1(s4!)
1c00c526:	fefa68e3          	bltu	s4,a5,1c00c516 <pos_libc_prf+0x258>
1c00c52a:	4782                	lw	a5,0(sp)
1c00c52c:	4732                	lw	a4,12(sp)
1c00c52e:	40f90933          	sub	s2,s2,a5
1c00c532:	3e0710e3          	bnez	a4,1c00d112 <pos_libc_prf+0xe54>
1c00c536:	47c2                	lw	a5,16(sp)
1c00c538:	3c079de3          	bnez	a5,1c00d112 <pos_libc_prf+0xe54>
1c00c53c:	4712                	lw	a4,4(sp)
1c00c53e:	837d                	srli	a4,a4,0x1f
1c00c540:	c23a                	sw	a4,4(sp)
1c00c542:	3404c463          	bltz	s1,1c00c88a <pos_libc_prf+0x5cc>
1c00c546:	4712                	lw	a4,4(sp)
1c00c548:	412484b3          	sub	s1,s1,s2
1c00c54c:	1040                	addi	s0,sp,36
1c00c54e:	94ba                	add	s1,s1,a4
1c00c550:	0404e4b3          	p.max	s1,s1,zero
1c00c554:	009907b3          	add	a5,s2,s1
1c00c558:	40fb0b33          	sub	s6,s6,a5
1c00c55c:	4c81                	li	s9,0
1c00c55e:	cc02                	sw	zero,24(sp)
1c00c560:	4a01                	li	s4,0
1c00c562:	4722                	lw	a4,8(sp)
1c00c564:	e305                	bnez	a4,1c00c584 <pos_libc_prf+0x2c6>
1c00c566:	01605f63          	blez	s6,1c00c584 <pos_libc_prf+0x2c6>
1c00c56a:	fffb0d93          	addi	s11,s6,-1
1c00c56e:	85de                	mv	a1,s7
1c00c570:	02000513          	li	a0,32
1c00c574:	9982                	jalr	s3
1c00c576:	1dfd                	addi	s11,s11,-1
1c00c578:	ebf52ee3          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c57c:	fffdb9e3          	p.bneimm	s11,-1,1c00c56e <pos_libc_prf+0x2b0>
1c00c580:	9ada                	add	s5,s5,s6
1c00c582:	5b7d                	li	s6,-1
1c00c584:	4792                	lw	a5,4(sp)
1c00c586:	00f40db3          	add	s11,s0,a5
1c00c58a:	a031                	j	1c00c596 <pos_libc_prf+0x2d8>
1c00c58c:	0014450b          	p.lbu	a0,1(s0!)
1c00c590:	9982                	jalr	s3
1c00c592:	ebf521e3          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c596:	85de                	mv	a1,s7
1c00c598:	ffb41ae3          	bne	s0,s11,1c00c58c <pos_libc_prf+0x2ce>
1c00c59c:	8da6                	mv	s11,s1
1c00c59e:	a021                	j	1c00c5a6 <pos_libc_prf+0x2e8>
1c00c5a0:	9982                	jalr	s3
1c00c5a2:	e9f529e3          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c5a6:	1dfd                	addi	s11,s11,-1
1c00c5a8:	85de                	mv	a1,s7
1c00c5aa:	03000513          	li	a0,48
1c00c5ae:	fffdb9e3          	p.bneimm	s11,-1,1c00c5a0 <pos_libc_prf+0x2e2>
1c00c5b2:	4712                	lw	a4,4(sp)
1c00c5b4:	40e90933          	sub	s2,s2,a4
1c00c5b8:	c44a                	sw	s2,8(sp)
1c00c5ba:	e40a15e3          	bnez	s4,1c00c404 <pos_libc_prf+0x146>
1c00c5be:	47e2                	lw	a5,24(sp)
1c00c5c0:	16079763          	bnez	a5,1c00c72e <pos_libc_prf+0x470>
1c00c5c4:	120c8863          	beqz	s9,1c00c6f4 <pos_libc_prf+0x436>
1c00c5c8:	00044783          	lbu	a5,0(s0)
1c00c5cc:	4a25                	li	s4,9
1c00c5ce:	02e00d93          	li	s11,46
1c00c5d2:	a801                	j	1c00c5e2 <pos_libc_prf+0x324>
1c00c5d4:	0405                	addi	s0,s0,1
1c00c5d6:	9982                	jalr	s3
1c00c5d8:	e5f52ee3          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c5dc:	00044783          	lbu	a5,0(s0)
1c00c5e0:	197d                	addi	s2,s2,-1
1c00c5e2:	fd078713          	addi	a4,a5,-48
1c00c5e6:	853e                	mv	a0,a5
1c00c5e8:	85de                	mv	a1,s7
1c00c5ea:	feea75e3          	bleu	a4,s4,1c00c5d4 <pos_libc_prf+0x316>
1c00c5ee:	ffb783e3          	beq	a5,s11,1c00c5d4 <pos_libc_prf+0x316>
1c00c5f2:	8a66                	mv	s4,s9
1c00c5f4:	85de                	mv	a1,s7
1c00c5f6:	03000513          	li	a0,48
1c00c5fa:	01404463          	bgtz	s4,1c00c602 <pos_libc_prf+0x344>
1c00c5fe:	0070106f          	j	1c00de04 <pos_libc_prf+0x1b46>
1c00c602:	9982                	jalr	s3
1c00c604:	1a7d                	addi	s4,s4,-1
1c00c606:	fff537e3          	p.bneimm	a0,-1,1c00c5f4 <pos_libc_prf+0x336>
1c00c60a:	547d                	li	s0,-1
1c00c60c:	bbc1                	j	1c00c3dc <pos_libc_prf+0x11e>
1c00c60e:	000d4d83          	lbu	s11,0(s10)
1c00c612:	02a00793          	li	a5,42
1c00c616:	001d0513          	addi	a0,s10,1
1c00c61a:	08fd8e63          	beq	s11,a5,1c00c6b6 <pos_libc_prf+0x3f8>
1c00c61e:	fd0d8793          	addi	a5,s11,-48
1c00c622:	4825                	li	a6,9
1c00c624:	846e                	mv	s0,s11
1c00c626:	86aa                	mv	a3,a0
1c00c628:	4481                	li	s1,0
1c00c62a:	45a5                	li	a1,9
1c00c62c:	8d2a                	mv	s10,a0
1c00c62e:	d4f86ee3          	bltu	a6,a5,1c00c38a <pos_libc_prf+0xcc>
1c00c632:	00249793          	slli	a5,s1,0x2
1c00c636:	97a6                	add	a5,a5,s1
1c00c638:	0786                	slli	a5,a5,0x1
1c00c63a:	97ee                	add	a5,a5,s11
1c00c63c:	0016cd8b          	p.lbu	s11,1(a3!) # 1a10f001 <__l1_end+0xa10e67d>
1c00c640:	fd078493          	addi	s1,a5,-48
1c00c644:	fd0d8793          	addi	a5,s11,-48
1c00c648:	846e                	mv	s0,s11
1c00c64a:	fef5f4e3          	bleu	a5,a1,1c00c632 <pos_libc_prf+0x374>
1c00c64e:	8d36                	mv	s10,a3
1c00c650:	bb2d                	j	1c00c38a <pos_libc_prf+0xcc>
1c00c652:	000a2b03          	lw	s6,0(s4)
1c00c656:	0a11                	addi	s4,s4,4
1c00c658:	080b4363          	bltz	s6,1c00c6de <pos_libc_prf+0x420>
1c00c65c:	00094d83          	lbu	s11,0(s2)
1c00c660:	00240d13          	addi	s10,s0,2
1c00c664:	bb29                	j	1c00c37e <pos_libc_prf+0xc0>
1c00c666:	07500693          	li	a3,117
1c00c66a:	10dd8763          	beq	s11,a3,1c00c778 <pos_libc_prf+0x4ba>
1c00c66e:	07800693          	li	a3,120
1c00c672:	10dd8363          	beq	s11,a3,1c00c778 <pos_libc_prf+0x4ba>
1c00c676:	07300793          	li	a5,115
1c00c67a:	0cfd96e3          	bne	s11,a5,1c00cf46 <pos_libc_prf+0xc88>
1c00c67e:	004a0713          	addi	a4,s4,4
1c00c682:	ce3a                	sw	a4,28(sp)
1c00c684:	000a2403          	lw	s0,0(s4)
1c00c688:	3c04c5e3          	bltz	s1,1c00d252 <pos_libc_prf+0xf94>
1c00c68c:	460484e3          	beqz	s1,1c00d2f4 <pos_libc_prf+0x1036>
1c00c690:	00044783          	lbu	a5,0(s0)
1c00c694:	44078fe3          	beqz	a5,1c00d2f2 <pos_libc_prf+0x1034>
1c00c698:	00140793          	addi	a5,s0,1
1c00c69c:	4901                	li	s2,0
1c00c69e:	a021                	j	1c00c6a6 <pos_libc_prf+0x3e8>
1c00c6a0:	0017c70b          	p.lbu	a4,1(a5!)
1c00c6a4:	c701                	beqz	a4,1c00c6ac <pos_libc_prf+0x3ee>
1c00c6a6:	0905                	addi	s2,s2,1
1c00c6a8:	fe994ce3          	blt	s2,s1,1c00c6a0 <pos_libc_prf+0x3e2>
1c00c6ac:	412b0b33          	sub	s6,s6,s2
1c00c6b0:	4481                	li	s1,0
1c00c6b2:	c202                	sw	zero,4(sp)
1c00c6b4:	b565                	j	1c00c55c <pos_libc_prf+0x29e>
1c00c6b6:	001d4d83          	lbu	s11,1(s10)
1c00c6ba:	000a2483          	lw	s1,0(s4)
1c00c6be:	0d09                	addi	s10,s10,2
1c00c6c0:	846e                	mv	s0,s11
1c00c6c2:	0a11                	addi	s4,s4,4
1c00c6c4:	b1d9                	j	1c00c38a <pos_libc_prf+0xcc>
1c00c6c6:	02500793          	li	a5,37
1c00c6ca:	06fd9ee3          	bne	s11,a5,1c00cf46 <pos_libc_prf+0xc88>
1c00c6ce:	85de                	mv	a1,s7
1c00c6d0:	02500513          	li	a0,37
1c00c6d4:	9982                	jalr	s3
1c00c6d6:	d5f52fe3          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c6da:	0a85                	addi	s5,s5,1
1c00c6dc:	b105                	j	1c00c2fc <pos_libc_prf+0x3e>
1c00c6de:	4785                	li	a5,1
1c00c6e0:	41600b33          	neg	s6,s6
1c00c6e4:	c43e                	sw	a5,8(sp)
1c00c6e6:	bf9d                	j	1c00c65c <pos_libc_prf+0x39e>
1c00c6e8:	0014450b          	p.lbu	a0,1(s0!)
1c00c6ec:	197d                	addi	s2,s2,-1
1c00c6ee:	9982                	jalr	s3
1c00c6f0:	d5f522e3          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00c6f4:	85de                	mv	a1,s7
1c00c6f6:	ff2049e3          	bgtz	s2,1c00c6e8 <pos_libc_prf+0x42a>
1c00c6fa:	4792                	lw	a5,4(sp)
1c00c6fc:	4a72                	lw	s4,28(sp)
1c00c6fe:	00fa8733          	add	a4,s5,a5
1c00c702:	47a2                	lw	a5,8(sp)
1c00c704:	9726                	add	a4,a4,s1
1c00c706:	00e78ab3          	add	s5,a5,a4
1c00c70a:	bf6059e3          	blez	s6,1c00c2fc <pos_libc_prf+0x3e>
1c00c70e:	9ada                	add	s5,s5,s6
1c00c710:	1b7d                	addi	s6,s6,-1
1c00c712:	a029                	j	1c00c71c <pos_libc_prf+0x45e>
1c00c714:	01fb3463          	p.bneimm	s6,-1,1c00c71c <pos_libc_prf+0x45e>
1c00c718:	6040106f          	j	1c00dd1c <pos_libc_prf+0x1a5e>
1c00c71c:	85de                	mv	a1,s7
1c00c71e:	02000513          	li	a0,32
1c00c722:	9982                	jalr	s3
1c00c724:	1b7d                	addi	s6,s6,-1
1c00c726:	fff537e3          	p.bneimm	a0,-1,1c00c714 <pos_libc_prf+0x456>
1c00c72a:	547d                	li	s0,-1
1c00c72c:	b945                	j	1c00c3dc <pos_libc_prf+0x11e>
1c00c72e:	02e00a13          	li	s4,46
1c00c732:	a029                	j	1c00c73c <pos_libc_prf+0x47e>
1c00c734:	014d9463          	bne	s11,s4,1c00c73c <pos_libc_prf+0x47e>
1c00c738:	53a0106f          	j	1c00dc72 <pos_libc_prf+0x19b4>
1c00c73c:	00144d8b          	p.lbu	s11,1(s0!)
1c00c740:	85de                	mv	a1,s7
1c00c742:	197d                	addi	s2,s2,-1
1c00c744:	856e                	mv	a0,s11
1c00c746:	9982                	jalr	s3
1c00c748:	fff536e3          	p.bneimm	a0,-1,1c00c734 <pos_libc_prf+0x476>
1c00c74c:	547d                	li	s0,-1
1c00c74e:	b179                	j	1c00c3dc <pos_libc_prf+0x11e>
1c00c750:	188d86e3          	beq	s11,s0,1c00d0dc <pos_libc_prf+0xe1e>
1c00c754:	8d36                	mv	s10,a3
1c00c756:	b185                	j	1c00c3b6 <pos_libc_prf+0xf8>
1c00c758:	188d89e3          	beq	s11,s0,1c00d0ea <pos_libc_prf+0xe2c>
1c00c75c:	8d36                	mv	s10,a3
1c00c75e:	06800793          	li	a5,104
1c00c762:	b991                	j	1c00c3b6 <pos_libc_prf+0xf8>
1c00c764:	06300693          	li	a3,99
1c00c768:	70dd8a63          	beq	s11,a3,1c00ce7c <pos_libc_prf+0xbbe>
1c00c76c:	d3b6e5e3          	bltu	a3,s11,1c00c496 <pos_libc_prf+0x1d8>
1c00c770:	05800693          	li	a3,88
1c00c774:	7cdd9963          	bne	s11,a3,1c00cf46 <pos_libc_prf+0xc88>
1c00c778:	06c00693          	li	a3,108
1c00c77c:	18d783e3          	beq	a5,a3,1c00d102 <pos_libc_prf+0xe44>
1c00c780:	07a00693          	li	a3,122
1c00c784:	16d78fe3          	beq	a5,a3,1c00d102 <pos_libc_prf+0xe44>
1c00c788:	04c00693          	li	a3,76
1c00c78c:	2ed781e3          	beq	a5,a3,1c00d26e <pos_libc_prf+0xfb0>
1c00c790:	000a2403          	lw	s0,0(s4)
1c00c794:	0a11                	addi	s4,s4,4
1c00c796:	ce52                	sw	s4,28(sp)
1c00c798:	4c81                	li	s9,0
1c00c79a:	06f00793          	li	a5,111
1c00c79e:	3cfd84e3          	beq	s11,a5,1c00d366 <pos_libc_prf+0x10a8>
1c00c7a2:	07500793          	li	a5,117
1c00c7a6:	02410a13          	addi	s4,sp,36
1c00c7aa:	04fd8763          	beq	s11,a5,1c00c7f8 <pos_libc_prf+0x53a>
1c00c7ae:	4712                	lw	a4,4(sp)
1c00c7b0:	2a070be3          	beqz	a4,1c00d266 <pos_libc_prf+0xfa8>
1c00c7b4:	77e1                	lui	a5,0xffff8
1c00c7b6:	8307c793          	xori	a5,a5,-2000
1c00c7ba:	02f11223          	sh	a5,36(sp)
1c00c7be:	4909                	li	s2,2
1c00c7c0:	02610693          	addi	a3,sp,38
1c00c7c4:	87b6                	mv	a5,a3
1c00c7c6:	4e25                	li	t3,9
1c00c7c8:	f6443733          	p.bclr	a4,s0,27,4
1c00c7cc:	01cc9613          	slli	a2,s9,0x1c
1c00c7d0:	8011                	srli	s0,s0,0x4
1c00c7d2:	8c51                	or	s0,s0,a2
1c00c7d4:	004cdc93          	srli	s9,s9,0x4
1c00c7d8:	05770813          	addi	a6,a4,87
1c00c7dc:	03070593          	addi	a1,a4,48
1c00c7e0:	01946533          	or	a0,s0,s9
1c00c7e4:	06ee7663          	bleu	a4,t3,1c00c850 <pos_libc_prf+0x592>
1c00c7e8:	01078023          	sb	a6,0(a5) # ffff8000 <pulp__FC+0xffff8001>
1c00c7ec:	00178593          	addi	a1,a5,1
1c00c7f0:	c52d                	beqz	a0,1c00c85a <pos_libc_prf+0x59c>
1c00c7f2:	87ae                	mv	a5,a1
1c00c7f4:	bfd1                	j	1c00c7c8 <pos_libc_prf+0x50a>
1c00c7f6:	8a4a                	mv	s4,s2
1c00c7f8:	4629                	li	a2,10
1c00c7fa:	4681                	li	a3,0
1c00c7fc:	8522                	mv	a0,s0
1c00c7fe:	85e6                	mv	a1,s9
1c00c800:	f00fc0ef          	jal	ra,1c008f00 <__umoddi3>
1c00c804:	892a                	mv	s2,a0
1c00c806:	85e6                	mv	a1,s9
1c00c808:	8522                	mv	a0,s0
1c00c80a:	4629                	li	a2,10
1c00c80c:	4681                	li	a3,0
1c00c80e:	bb6fc0ef          	jal	ra,1c008bc4 <__udivdi3>
1c00c812:	03090913          	addi	s2,s2,48
1c00c816:	012a0023          	sb	s2,0(s4)
1c00c81a:	00a5e7b3          	or	a5,a1,a0
1c00c81e:	001a0913          	addi	s2,s4,1
1c00c822:	842a                	mv	s0,a0
1c00c824:	8cae                	mv	s9,a1
1c00c826:	fbe1                	bnez	a5,1c00c7f6 <pos_libc_prf+0x538>
1c00c828:	4782                	lw	a5,0(sp)
1c00c82a:	00090023          	sb	zero,0(s2)
1c00c82e:	40f90933          	sub	s2,s2,a5
1c00c832:	0147fd63          	bleu	s4,a5,1c00c84c <pos_libc_prf+0x58e>
1c00c836:	105c                	addi	a5,sp,36
1c00c838:	0007c683          	lbu	a3,0(a5)
1c00c83c:	000a4703          	lbu	a4,0(s4)
1c00c840:	feda0fab          	p.sb	a3,-1(s4!)
1c00c844:	00e780ab          	p.sb	a4,1(a5!)
1c00c848:	ff47e8e3          	bltu	a5,s4,1c00c838 <pos_libc_prf+0x57a>
1c00c84c:	c202                	sw	zero,4(sp)
1c00c84e:	b9d5                	j	1c00c542 <pos_libc_prf+0x284>
1c00c850:	00b78023          	sb	a1,0(a5)
1c00c854:	00178593          	addi	a1,a5,1
1c00c858:	fd49                	bnez	a0,1c00c7f2 <pos_libc_prf+0x534>
1c00c85a:	00058023          	sb	zero,0(a1) # 1a10f000 <__l1_end+0xa10e67c>
1c00c85e:	8d95                	sub	a1,a1,a3
1c00c860:	00f6fc63          	bleu	a5,a3,1c00c878 <pos_libc_prf+0x5ba>
1c00c864:	0006c603          	lbu	a2,0(a3)
1c00c868:	0007c703          	lbu	a4,0(a5)
1c00c86c:	fec78fab          	p.sb	a2,-1(a5!)
1c00c870:	00e680ab          	p.sb	a4,1(a3!)
1c00c874:	fef6e8e3          	bltu	a3,a5,1c00c864 <pos_libc_prf+0x5a6>
1c00c878:	05800793          	li	a5,88
1c00c87c:	28fd85e3          	beq	s11,a5,1c00d306 <pos_libc_prf+0x1048>
1c00c880:	4712                	lw	a4,4(sp)
1c00c882:	992e                	add	s2,s2,a1
1c00c884:	0706                	slli	a4,a4,0x1
1c00c886:	c23a                	sw	a4,4(sp)
1c00c888:	b96d                	j	1c00c542 <pos_libc_prf+0x284>
1c00c88a:	4c81                	li	s9,0
1c00c88c:	cc02                	sw	zero,24(sp)
1c00c88e:	4a01                	li	s4,0
1c00c890:	47d2                	lw	a5,20(sp)
1c00c892:	5c078f63          	beqz	a5,1c00ce70 <pos_libc_prf+0xbb2>
1c00c896:	412b04b3          	sub	s1,s6,s2
1c00c89a:	0404e4b3          	p.max	s1,s1,zero
1c00c89e:	009907b3          	add	a5,s2,s1
1c00c8a2:	40fb0b33          	sub	s6,s6,a5
1c00c8a6:	1040                	addi	s0,sp,36
1c00c8a8:	b96d                	j	1c00c562 <pos_libc_prf+0x2a4>
1c00c8aa:	4301                	li	t1,0
1c00c8ac:	4881                	li	a7,0
1c00c8ae:	4581                	li	a1,0
1c00c8b0:	4801                	li	a6,0
1c00c8b2:	4691                	li	a3,4
1c00c8b4:	411686b3          	sub	a3,a3,a7
1c00c8b8:	0136c0fb          	lp.setup	x1,a3,1c00c8de <pos_libc_prf+0x620>
1c00c8bc:	01f81713          	slli	a4,a6,0x1f
1c00c8c0:	0015d793          	srli	a5,a1,0x1
1c00c8c4:	8fd9                	or	a5,a5,a4
1c00c8c6:	fc15b5b3          	p.bclr	a1,a1,30,1
1c00c8ca:	00f58f33          	add	t5,a1,a5
1c00c8ce:	00185813          	srli	a6,a6,0x1
1c00c8d2:	00bf35b3          	sltu	a1,t5,a1
1c00c8d6:	01058533          	add	a0,a1,a6
1c00c8da:	882a                	mv	a6,a0
1c00c8dc:	85fa                	mv	a1,t5
1c00c8de:	0885                	addi	a7,a7,1
1c00c8e0:	0004d363          	bgez	s1,1c00c8e6 <pos_libc_prf+0x628>
1c00c8e4:	4499                	li	s1,6
1c00c8e6:	04700793          	li	a5,71
1c00c8ea:	0cfe8de3          	beq	t4,a5,1c00d1c4 <pos_libc_prf+0xf06>
1c00c8ee:	06600793          	li	a5,102
1c00c8f2:	4801                	li	a6,0
1c00c8f4:	0efd86e3          	beq	s11,a5,1c00d1e0 <pos_libc_prf+0xf22>
1c00c8f8:	00148e93          	addi	t4,s1,1
1c00c8fc:	4741                	li	a4,16
1c00c8fe:	04eeceb3          	p.min	t4,t4,a4
1c00c902:	1efd                	addi	t4,t4,-1
1c00c904:	4281                	li	t0,0
1c00c906:	080006b7          	lui	a3,0x8000
1c00c90a:	4f95                	li	t6,5
1c00c90c:	0e85                	addi	t4,t4,1
1c00c90e:	02fec0fb          	lp.setup	x1,t4,1c00c96c <pos_libc_prf+0x6ae>
1c00c912:	00228713          	addi	a4,t0,2
1c00c916:	005737b3          	sltu	a5,a4,t0
1c00c91a:	97b6                	add	a5,a5,a3
1c00c91c:	03f7d5b3          	divu	a1,a5,t6
1c00c920:	00375613          	srli	a2,a4,0x3
1c00c924:	00259693          	slli	a3,a1,0x2
1c00c928:	96ae                	add	a3,a3,a1
1c00c92a:	8f95                	sub	a5,a5,a3
1c00c92c:	07f6                	slli	a5,a5,0x1d
1c00c92e:	8e5d                	or	a2,a2,a5
1c00c930:	03f65633          	divu	a2,a2,t6
1c00c934:	00361693          	slli	a3,a2,0x3
1c00c938:	00561793          	slli	a5,a2,0x5
1c00c93c:	97b6                	add	a5,a5,a3
1c00c93e:	40f707b3          	sub	a5,a4,a5
1c00c942:	03f7d7b3          	divu	a5,a5,t6
1c00c946:	8275                	srli	a2,a2,0x1d
1c00c948:	962e                	add	a2,a2,a1
1c00c94a:	96be                	add	a3,a3,a5
1c00c94c:	00f6b7b3          	sltu	a5,a3,a5
1c00c950:	00c78733          	add	a4,a5,a2
1c00c954:	01f71613          	slli	a2,a4,0x1f
1c00c958:	0016d793          	srli	a5,a3,0x1
1c00c95c:	8fd1                	or	a5,a5,a2
1c00c95e:	fc16b6b3          	p.bclr	a3,a3,30,1
1c00c962:	00f682b3          	add	t0,a3,a5
1c00c966:	8305                	srli	a4,a4,0x1
1c00c968:	00d2b6b3          	sltu	a3,t0,a3
1c00c96c:	96ba                	add	a3,a3,a4
1c00c96e:	005f07b3          	add	a5,t5,t0
1c00c972:	01e7bf33          	sltu	t5,a5,t5
1c00c976:	96aa                	add	a3,a3,a0
1c00c978:	96fa                	add	a3,a3,t5
1c00c97a:	f606b5b3          	p.bclr	a1,a3,27,0
1c00c97e:	cdb9                	beqz	a1,1c00c9dc <pos_libc_prf+0x71e>
1c00c980:	00278513          	addi	a0,a5,2
1c00c984:	00f537b3          	sltu	a5,a0,a5
1c00c988:	97b6                	add	a5,a5,a3
1c00c98a:	4f15                	li	t5,5
1c00c98c:	03e7deb3          	divu	t4,a5,t5
1c00c990:	00355693          	srli	a3,a0,0x3
1c00c994:	0305                	addi	t1,t1,1
1c00c996:	002e9593          	slli	a1,t4,0x2
1c00c99a:	95f6                	add	a1,a1,t4
1c00c99c:	8f8d                	sub	a5,a5,a1
1c00c99e:	07f6                	slli	a5,a5,0x1d
1c00c9a0:	8edd                	or	a3,a3,a5
1c00c9a2:	03e6d6b3          	divu	a3,a3,t5
1c00c9a6:	00369593          	slli	a1,a3,0x3
1c00c9aa:	00569793          	slli	a5,a3,0x5
1c00c9ae:	97ae                	add	a5,a5,a1
1c00c9b0:	40f507b3          	sub	a5,a0,a5
1c00c9b4:	03e7d7b3          	divu	a5,a5,t5
1c00c9b8:	82f5                	srli	a3,a3,0x1d
1c00c9ba:	96f6                	add	a3,a3,t4
1c00c9bc:	95be                	add	a1,a1,a5
1c00c9be:	00f5b7b3          	sltu	a5,a1,a5
1c00c9c2:	96be                	add	a3,a3,a5
1c00c9c4:	01f69513          	slli	a0,a3,0x1f
1c00c9c8:	0015d793          	srli	a5,a1,0x1
1c00c9cc:	8fc9                	or	a5,a5,a0
1c00c9ce:	fc15b5b3          	p.bclr	a1,a1,30,1
1c00c9d2:	97ae                	add	a5,a5,a1
1c00c9d4:	8285                	srli	a3,a3,0x1
1c00c9d6:	00b7b5b3          	sltu	a1,a5,a1
1c00c9da:	96ae                	add	a3,a3,a1
1c00c9dc:	06600593          	li	a1,102
1c00c9e0:	001e0513          	addi	a0,t3,1 # 400001 <__l1_heap_size+0x300985>
1c00c9e4:	1ebd87e3          	beq	s11,a1,1c00d3d2 <pos_libc_prf+0x1114>
1c00c9e8:	01e7de93          	srli	t4,a5,0x1e
1c00c9ec:	00279593          	slli	a1,a5,0x2
1c00c9f0:	00269893          	slli	a7,a3,0x2
1c00c9f4:	97ae                	add	a5,a5,a1
1c00c9f6:	011ee8b3          	or	a7,t4,a7
1c00c9fa:	00b7b5b3          	sltu	a1,a5,a1
1c00c9fe:	96c6                	add	a3,a3,a7
1c00ca00:	96ae                	add	a3,a3,a1
1c00ca02:	0686                	slli	a3,a3,0x1
1c00ca04:	01f7d593          	srli	a1,a5,0x1f
1c00ca08:	8ecd                	or	a3,a3,a1
1c00ca0a:	01c6d593          	srli	a1,a3,0x1c
1c00ca0e:	4712                	lw	a4,4(sp)
1c00ca10:	88ae                	mv	a7,a1
1c00ca12:	03058593          	addi	a1,a1,48
1c00ca16:	011038b3          	snez	a7,a7
1c00ca1a:	00be0023          	sb	a1,0(t3)
1c00ca1e:	c7c6beb3          	p.bclr	t4,a3,3,28
1c00ca22:	00179593          	slli	a1,a5,0x1
1c00ca26:	41130333          	sub	t1,t1,a7
1c00ca2a:	0a0719e3          	bnez	a4,1c00d2dc <pos_libc_prf+0x101e>
1c00ca2e:	8ca6                	mv	s9,s1
1c00ca30:	3c905d63          	blez	s1,1c00ce0a <pos_libc_prf+0xb4c>
1c00ca34:	02e00693          	li	a3,46
1c00ca38:	002e0793          	addi	a5,t3,2
1c00ca3c:	00de00a3          	sb	a3,1(t3)
1c00ca40:	00259693          	slli	a3,a1,0x2
1c00ca44:	01e5d893          	srli	a7,a1,0x1e
1c00ca48:	002e9513          	slli	a0,t4,0x2
1c00ca4c:	95b6                	add	a1,a1,a3
1c00ca4e:	00a8e533          	or	a0,a7,a0
1c00ca52:	00d5b8b3          	sltu	a7,a1,a3
1c00ca56:	01d506b3          	add	a3,a0,t4
1c00ca5a:	98b6                	add	a7,a7,a3
1c00ca5c:	0886                	slli	a7,a7,0x1
1c00ca5e:	01f5d693          	srli	a3,a1,0x1f
1c00ca62:	0116e8b3          	or	a7,a3,a7
1c00ca66:	01c8de13          	srli	t3,a7,0x1c
1c00ca6a:	030e0e13          	addi	t3,t3,48
1c00ca6e:	100006b7          	lui	a3,0x10000
1c00ca72:	fff68513          	addi	a0,a3,-1 # fffffff <__l1_heap_size+0xff00983>
1c00ca76:	01c78023          	sb	t3,0(a5)
1c00ca7a:	00159e93          	slli	t4,a1,0x1
1c00ca7e:	00a8f8b3          	and	a7,a7,a0
1c00ca82:	0014b463          	p.bneimm	s1,1,1c00ca8a <pos_libc_prf+0x7cc>
1c00ca86:	3de0106f          	j	1c00de64 <pos_libc_prf+0x1ba6>
1c00ca8a:	00359693          	slli	a3,a1,0x3
1c00ca8e:	00289e13          	slli	t3,a7,0x2
1c00ca92:	01eedf13          	srli	t5,t4,0x1e
1c00ca96:	01d685b3          	add	a1,a3,t4
1c00ca9a:	01cf6e33          	or	t3,t5,t3
1c00ca9e:	98f2                	add	a7,a7,t3
1c00caa0:	00d5b6b3          	sltu	a3,a1,a3
1c00caa4:	96c6                	add	a3,a3,a7
1c00caa6:	0686                	slli	a3,a3,0x1
1c00caa8:	01f5d893          	srli	a7,a1,0x1f
1c00caac:	00d8e6b3          	or	a3,a7,a3
1c00cab0:	01c6d893          	srli	a7,a3,0x1c
1c00cab4:	03088893          	addi	a7,a7,48
1c00cab8:	011780a3          	sb	a7,1(a5)
1c00cabc:	ffe48c93          	addi	s9,s1,-2
1c00cac0:	00159e13          	slli	t3,a1,0x1
1c00cac4:	8ee9                	and	a3,a3,a0
1c00cac6:	01904463          	bgtz	s9,1c00cace <pos_libc_prf+0x810>
1c00caca:	3920106f          	j	1c00de5c <pos_libc_prf+0x1b9e>
1c00cace:	01ee5e93          	srli	t4,t3,0x1e
1c00cad2:	058e                	slli	a1,a1,0x3
1c00cad4:	00269893          	slli	a7,a3,0x2
1c00cad8:	9e2e                	add	t3,t3,a1
1c00cada:	011ee8b3          	or	a7,t4,a7
1c00cade:	96c6                	add	a3,a3,a7
1c00cae0:	00be35b3          	sltu	a1,t3,a1
1c00cae4:	95b6                	add	a1,a1,a3
1c00cae6:	0586                	slli	a1,a1,0x1
1c00cae8:	01fe5693          	srli	a3,t3,0x1f
1c00caec:	8dd5                	or	a1,a1,a3
1c00caee:	01c5d693          	srli	a3,a1,0x1c
1c00caf2:	03068693          	addi	a3,a3,48
1c00caf6:	00d78123          	sb	a3,2(a5)
1c00cafa:	ffd48c93          	addi	s9,s1,-3
1c00cafe:	001e1e93          	slli	t4,t3,0x1
1c00cb02:	8de9                	and	a1,a1,a0
1c00cb04:	01904463          	bgtz	s9,1c00cb0c <pos_libc_prf+0x84e>
1c00cb08:	34c0106f          	j	1c00de54 <pos_libc_prf+0x1b96>
1c00cb0c:	003e1693          	slli	a3,t3,0x3
1c00cb10:	00259893          	slli	a7,a1,0x2
1c00cb14:	01eede13          	srli	t3,t4,0x1e
1c00cb18:	011e68b3          	or	a7,t3,a7
1c00cb1c:	9eb6                	add	t4,t4,a3
1c00cb1e:	95c6                	add	a1,a1,a7
1c00cb20:	00deb6b3          	sltu	a3,t4,a3
1c00cb24:	96ae                	add	a3,a3,a1
1c00cb26:	0686                	slli	a3,a3,0x1
1c00cb28:	01fed593          	srli	a1,t4,0x1f
1c00cb2c:	8ecd                	or	a3,a3,a1
1c00cb2e:	01c6d593          	srli	a1,a3,0x1c
1c00cb32:	03058593          	addi	a1,a1,48
1c00cb36:	00b781a3          	sb	a1,3(a5)
1c00cb3a:	ffc48c93          	addi	s9,s1,-4
1c00cb3e:	001e9e13          	slli	t3,t4,0x1
1c00cb42:	8ee9                	and	a3,a3,a0
1c00cb44:	01904463          	bgtz	s9,1c00cb4c <pos_libc_prf+0x88e>
1c00cb48:	3040106f          	j	1c00de4c <pos_libc_prf+0x1b8e>
1c00cb4c:	003e9593          	slli	a1,t4,0x3
1c00cb50:	00269893          	slli	a7,a3,0x2
1c00cb54:	01ee5e93          	srli	t4,t3,0x1e
1c00cb58:	011ee8b3          	or	a7,t4,a7
1c00cb5c:	9e2e                	add	t3,t3,a1
1c00cb5e:	96c6                	add	a3,a3,a7
1c00cb60:	00be35b3          	sltu	a1,t3,a1
1c00cb64:	95b6                	add	a1,a1,a3
1c00cb66:	0586                	slli	a1,a1,0x1
1c00cb68:	01fe5693          	srli	a3,t3,0x1f
1c00cb6c:	8dd5                	or	a1,a1,a3
1c00cb6e:	01c5d693          	srli	a3,a1,0x1c
1c00cb72:	03068693          	addi	a3,a3,48
1c00cb76:	00d78223          	sb	a3,4(a5)
1c00cb7a:	ffb48c93          	addi	s9,s1,-5
1c00cb7e:	001e1e93          	slli	t4,t3,0x1
1c00cb82:	8de9                	and	a1,a1,a0
1c00cb84:	01904463          	bgtz	s9,1c00cb8c <pos_libc_prf+0x8ce>
1c00cb88:	2bc0106f          	j	1c00de44 <pos_libc_prf+0x1b86>
1c00cb8c:	003e1693          	slli	a3,t3,0x3
1c00cb90:	00259893          	slli	a7,a1,0x2
1c00cb94:	01eede13          	srli	t3,t4,0x1e
1c00cb98:	011e68b3          	or	a7,t3,a7
1c00cb9c:	9eb6                	add	t4,t4,a3
1c00cb9e:	95c6                	add	a1,a1,a7
1c00cba0:	00deb6b3          	sltu	a3,t4,a3
1c00cba4:	96ae                	add	a3,a3,a1
1c00cba6:	0686                	slli	a3,a3,0x1
1c00cba8:	01fed593          	srli	a1,t4,0x1f
1c00cbac:	8ecd                	or	a3,a3,a1
1c00cbae:	01c6d593          	srli	a1,a3,0x1c
1c00cbb2:	03058593          	addi	a1,a1,48
1c00cbb6:	00b782a3          	sb	a1,5(a5)
1c00cbba:	ffa48c93          	addi	s9,s1,-6
1c00cbbe:	001e9e13          	slli	t3,t4,0x1
1c00cbc2:	8ee9                	and	a3,a3,a0
1c00cbc4:	01904463          	bgtz	s9,1c00cbcc <pos_libc_prf+0x90e>
1c00cbc8:	2740106f          	j	1c00de3c <pos_libc_prf+0x1b7e>
1c00cbcc:	003e9593          	slli	a1,t4,0x3
1c00cbd0:	00269893          	slli	a7,a3,0x2
1c00cbd4:	01ee5e93          	srli	t4,t3,0x1e
1c00cbd8:	011ee8b3          	or	a7,t4,a7
1c00cbdc:	9e2e                	add	t3,t3,a1
1c00cbde:	96c6                	add	a3,a3,a7
1c00cbe0:	00be35b3          	sltu	a1,t3,a1
1c00cbe4:	95b6                	add	a1,a1,a3
1c00cbe6:	0586                	slli	a1,a1,0x1
1c00cbe8:	01fe5693          	srli	a3,t3,0x1f
1c00cbec:	8dd5                	or	a1,a1,a3
1c00cbee:	01c5d693          	srli	a3,a1,0x1c
1c00cbf2:	03068693          	addi	a3,a3,48
1c00cbf6:	00d78323          	sb	a3,6(a5)
1c00cbfa:	ff948c93          	addi	s9,s1,-7
1c00cbfe:	001e1e93          	slli	t4,t3,0x1
1c00cc02:	8de9                	and	a1,a1,a0
1c00cc04:	01904463          	bgtz	s9,1c00cc0c <pos_libc_prf+0x94e>
1c00cc08:	22c0106f          	j	1c00de34 <pos_libc_prf+0x1b76>
1c00cc0c:	003e1693          	slli	a3,t3,0x3
1c00cc10:	00259893          	slli	a7,a1,0x2
1c00cc14:	01eede13          	srli	t3,t4,0x1e
1c00cc18:	011e68b3          	or	a7,t3,a7
1c00cc1c:	9eb6                	add	t4,t4,a3
1c00cc1e:	95c6                	add	a1,a1,a7
1c00cc20:	00deb6b3          	sltu	a3,t4,a3
1c00cc24:	96ae                	add	a3,a3,a1
1c00cc26:	0686                	slli	a3,a3,0x1
1c00cc28:	01fed593          	srli	a1,t4,0x1f
1c00cc2c:	8ecd                	or	a3,a3,a1
1c00cc2e:	01c6d593          	srli	a1,a3,0x1c
1c00cc32:	03058593          	addi	a1,a1,48
1c00cc36:	00b783a3          	sb	a1,7(a5)
1c00cc3a:	ff848c93          	addi	s9,s1,-8
1c00cc3e:	001e9e13          	slli	t3,t4,0x1
1c00cc42:	8ee9                	and	a3,a3,a0
1c00cc44:	01904463          	bgtz	s9,1c00cc4c <pos_libc_prf+0x98e>
1c00cc48:	1e40106f          	j	1c00de2c <pos_libc_prf+0x1b6e>
1c00cc4c:	003e9593          	slli	a1,t4,0x3
1c00cc50:	00269893          	slli	a7,a3,0x2
1c00cc54:	01ee5e93          	srli	t4,t3,0x1e
1c00cc58:	011ee8b3          	or	a7,t4,a7
1c00cc5c:	9e2e                	add	t3,t3,a1
1c00cc5e:	96c6                	add	a3,a3,a7
1c00cc60:	00be35b3          	sltu	a1,t3,a1
1c00cc64:	95b6                	add	a1,a1,a3
1c00cc66:	0586                	slli	a1,a1,0x1
1c00cc68:	01fe5693          	srli	a3,t3,0x1f
1c00cc6c:	8dd5                	or	a1,a1,a3
1c00cc6e:	01c5d693          	srli	a3,a1,0x1c
1c00cc72:	03068693          	addi	a3,a3,48
1c00cc76:	00d78423          	sb	a3,8(a5)
1c00cc7a:	ff748c93          	addi	s9,s1,-9
1c00cc7e:	001e1e93          	slli	t4,t3,0x1
1c00cc82:	8de9                	and	a1,a1,a0
1c00cc84:	01904463          	bgtz	s9,1c00cc8c <pos_libc_prf+0x9ce>
1c00cc88:	19c0106f          	j	1c00de24 <pos_libc_prf+0x1b66>
1c00cc8c:	003e1693          	slli	a3,t3,0x3
1c00cc90:	00259893          	slli	a7,a1,0x2
1c00cc94:	01eedf13          	srli	t5,t4,0x1e
1c00cc98:	01d68e33          	add	t3,a3,t4
1c00cc9c:	011f68b3          	or	a7,t5,a7
1c00cca0:	98ae                	add	a7,a7,a1
1c00cca2:	00de36b3          	sltu	a3,t3,a3
1c00cca6:	96c6                	add	a3,a3,a7
1c00cca8:	01fe5593          	srli	a1,t3,0x1f
1c00ccac:	0686                	slli	a3,a3,0x1
1c00ccae:	8ecd                	or	a3,a3,a1
1c00ccb0:	01c6d593          	srli	a1,a3,0x1c
1c00ccb4:	03058593          	addi	a1,a1,48
1c00ccb8:	00b784a3          	sb	a1,9(a5)
1c00ccbc:	ff648c93          	addi	s9,s1,-10
1c00ccc0:	8ee9                	and	a3,a3,a0
1c00ccc2:	001e1593          	slli	a1,t3,0x1
1c00ccc6:	00a78513          	addi	a0,a5,10
1c00ccca:	15905063          	blez	s9,1c00ce0a <pos_libc_prf+0xb4c>
1c00ccce:	00259513          	slli	a0,a1,0x2
1c00ccd2:	01e5d893          	srli	a7,a1,0x1e
1c00ccd6:	00269e13          	slli	t3,a3,0x2
1c00ccda:	95aa                	add	a1,a1,a0
1c00ccdc:	01c8ee33          	or	t3,a7,t3
1c00cce0:	96f2                	add	a3,a3,t3
1c00cce2:	00a5b8b3          	sltu	a7,a1,a0
1c00cce6:	98b6                	add	a7,a7,a3
1c00cce8:	0886                	slli	a7,a7,0x1
1c00ccea:	01f5d693          	srli	a3,a1,0x1f
1c00ccee:	0116e8b3          	or	a7,a3,a7
1c00ccf2:	01c8de13          	srli	t3,a7,0x1c
1c00ccf6:	030e0e13          	addi	t3,t3,48
1c00ccfa:	100006b7          	lui	a3,0x10000
1c00ccfe:	fff68513          	addi	a0,a3,-1 # fffffff <__l1_heap_size+0xff00983>
1c00cd02:	01c78523          	sb	t3,10(a5)
1c00cd06:	ff548c93          	addi	s9,s1,-11
1c00cd0a:	00159e93          	slli	t4,a1,0x1
1c00cd0e:	00a8f8b3          	and	a7,a7,a0
1c00cd12:	01904463          	bgtz	s9,1c00cd1a <pos_libc_prf+0xa5c>
1c00cd16:	1060106f          	j	1c00de1c <pos_libc_prf+0x1b5e>
1c00cd1a:	00359693          	slli	a3,a1,0x3
1c00cd1e:	00289e13          	slli	t3,a7,0x2
1c00cd22:	01eedf13          	srli	t5,t4,0x1e
1c00cd26:	01d685b3          	add	a1,a3,t4
1c00cd2a:	01cf6e33          	or	t3,t5,t3
1c00cd2e:	98f2                	add	a7,a7,t3
1c00cd30:	00d5b6b3          	sltu	a3,a1,a3
1c00cd34:	96c6                	add	a3,a3,a7
1c00cd36:	0686                	slli	a3,a3,0x1
1c00cd38:	01f5d893          	srli	a7,a1,0x1f
1c00cd3c:	00d8e6b3          	or	a3,a7,a3
1c00cd40:	01c6d893          	srli	a7,a3,0x1c
1c00cd44:	03088893          	addi	a7,a7,48
1c00cd48:	011785a3          	sb	a7,11(a5)
1c00cd4c:	ff448c93          	addi	s9,s1,-12
1c00cd50:	00159e13          	slli	t3,a1,0x1
1c00cd54:	8ee9                	and	a3,a3,a0
1c00cd56:	01904463          	bgtz	s9,1c00cd5e <pos_libc_prf+0xaa0>
1c00cd5a:	0ba0106f          	j	1c00de14 <pos_libc_prf+0x1b56>
1c00cd5e:	058e                	slli	a1,a1,0x3
1c00cd60:	01ee5e93          	srli	t4,t3,0x1e
1c00cd64:	00269893          	slli	a7,a3,0x2
1c00cd68:	9e2e                	add	t3,t3,a1
1c00cd6a:	011ee8b3          	or	a7,t4,a7
1c00cd6e:	96c6                	add	a3,a3,a7
1c00cd70:	00be35b3          	sltu	a1,t3,a1
1c00cd74:	95b6                	add	a1,a1,a3
1c00cd76:	0586                	slli	a1,a1,0x1
1c00cd78:	01fe5693          	srli	a3,t3,0x1f
1c00cd7c:	8dd5                	or	a1,a1,a3
1c00cd7e:	01c5d693          	srli	a3,a1,0x1c
1c00cd82:	03068693          	addi	a3,a3,48
1c00cd86:	00d78623          	sb	a3,12(a5)
1c00cd8a:	ff348c93          	addi	s9,s1,-13
1c00cd8e:	001e1893          	slli	a7,t3,0x1
1c00cd92:	01904463          	bgtz	s9,1c00cd9a <pos_libc_prf+0xadc>
1c00cd96:	0760106f          	j	1c00de0c <pos_libc_prf+0x1b4e>
1c00cd9a:	8de9                	and	a1,a1,a0
1c00cd9c:	003e1693          	slli	a3,t3,0x3
1c00cda0:	01e8de93          	srli	t4,a7,0x1e
1c00cda4:	00259e13          	slli	t3,a1,0x2
1c00cda8:	98b6                	add	a7,a7,a3
1c00cdaa:	01ceee33          	or	t3,t4,t3
1c00cdae:	95f2                	add	a1,a1,t3
1c00cdb0:	00d8b6b3          	sltu	a3,a7,a3
1c00cdb4:	96ae                	add	a3,a3,a1
1c00cdb6:	0686                	slli	a3,a3,0x1
1c00cdb8:	01f8d593          	srli	a1,a7,0x1f
1c00cdbc:	8ecd                	or	a3,a3,a1
1c00cdbe:	01c6d593          	srli	a1,a3,0x1c
1c00cdc2:	03058593          	addi	a1,a1,48
1c00cdc6:	00b786a3          	sb	a1,13(a5)
1c00cdca:	ff248c93          	addi	s9,s1,-14
1c00cdce:	8ee9                	and	a3,a3,a0
1c00cdd0:	00189593          	slli	a1,a7,0x1
1c00cdd4:	00e78513          	addi	a0,a5,14
1c00cdd8:	03905963          	blez	s9,1c00ce0a <pos_libc_prf+0xb4c>
1c00cddc:	00389513          	slli	a0,a7,0x3
1c00cde0:	01e5de13          	srli	t3,a1,0x1e
1c00cde4:	00269893          	slli	a7,a3,0x2
1c00cde8:	95aa                	add	a1,a1,a0
1c00cdea:	011e68b3          	or	a7,t3,a7
1c00cdee:	00a5b5b3          	sltu	a1,a1,a0
1c00cdf2:	96c6                	add	a3,a3,a7
1c00cdf4:	96ae                	add	a3,a3,a1
1c00cdf6:	0686                	slli	a3,a3,0x1
1c00cdf8:	82f1                	srli	a3,a3,0x1c
1c00cdfa:	03068693          	addi	a3,a3,48
1c00cdfe:	00f78513          	addi	a0,a5,15
1c00ce02:	00d78723          	sb	a3,14(a5)
1c00ce06:	ff148c93          	addi	s9,s1,-15
1c00ce0a:	44080a63          	beqz	a6,1c00d25e <pos_libc_prf+0xfa0>
1c00ce0e:	4581                	li	a1,0
1c00ce10:	cc02                	sw	zero,24(sp)
1c00ce12:	4a01                	li	s4,0
1c00ce14:	03000813          	li	a6,48
1c00ce18:	a011                	j	1c00ce1c <pos_libc_prf+0xb5e>
1c00ce1a:	853e                	mv	a0,a5
1c00ce1c:	fff50793          	addi	a5,a0,-1
1c00ce20:	0007c683          	lbu	a3,0(a5)
1c00ce24:	ff068be3          	beq	a3,a6,1c00ce1a <pos_libc_prf+0xb5c>
1c00ce28:	02e00813          	li	a6,46
1c00ce2c:	4c81                	li	s9,0
1c00ce2e:	01069363          	bne	a3,a6,1c00ce34 <pos_libc_prf+0xb76>
1c00ce32:	853e                	mv	a0,a5
1c00ce34:	0dfdf793          	andi	a5,s11,223
1c00ce38:	04500693          	li	a3,69
1c00ce3c:	3cd78363          	beq	a5,a3,1c00d202 <pos_libc_prf+0xf44>
1c00ce40:	4782                	lw	a5,0(sp)
1c00ce42:	00050023          	sb	zero,0(a0)
1c00ce46:	8d1d                	sub	a0,a0,a5
1c00ce48:	4732                	lw	a4,12(sp)
1c00ce4a:	2a071763          	bnez	a4,1c00d0f8 <pos_libc_prf+0xe3a>
1c00ce4e:	47c2                	lw	a5,16(sp)
1c00ce50:	2a079463          	bnez	a5,1c00d0f8 <pos_libc_prf+0xe3a>
1c00ce54:	02414783          	lbu	a5,36(sp)
1c00ce58:	02d00693          	li	a3,45
1c00ce5c:	c202                	sw	zero,4(sp)
1c00ce5e:	28d78d63          	beq	a5,a3,1c00d0f8 <pos_libc_prf+0xe3a>
1c00ce62:	fd078793          	addi	a5,a5,-48
1c00ce66:	46a5                	li	a3,9
1c00ce68:	00b50933          	add	s2,a0,a1
1c00ce6c:	a2f6f2e3          	bleu	a5,a3,1c00c890 <pos_libc_prf+0x5d2>
1c00ce70:	412b0b33          	sub	s6,s6,s2
1c00ce74:	1040                	addi	s0,sp,36
1c00ce76:	4481                	li	s1,0
1c00ce78:	eeaff06f          	j	1c00c562 <pos_libc_prf+0x2a4>
1c00ce7c:	000a2783          	lw	a5,0(s4)
1c00ce80:	0a11                	addi	s4,s4,4
1c00ce82:	ce52                	sw	s4,28(sp)
1c00ce84:	1b7d                	addi	s6,s6,-1
1c00ce86:	02f10223          	sb	a5,36(sp)
1c00ce8a:	4481                	li	s1,0
1c00ce8c:	c202                	sw	zero,4(sp)
1c00ce8e:	4905                	li	s2,1
1c00ce90:	1040                	addi	s0,sp,36
1c00ce92:	ecaff06f          	j	1c00c55c <pos_libc_prf+0x29e>
1c00ce96:	04c00613          	li	a2,76
1c00ce9a:	004a268b          	p.lw	a3,4(s4!)
1c00ce9e:	5cc783e3          	beq	a5,a2,1c00dc64 <pos_libc_prf+0x19a6>
1c00cea2:	28f65863          	ble	a5,a2,1c00d132 <pos_libc_prf+0xe74>
1c00cea6:	06800613          	li	a2,104
1c00ceaa:	00c78663          	beq	a5,a2,1c00ceb6 <pos_libc_prf+0xbf8>
1c00ceae:	0156a023          	sw	s5,0(a3)
1c00ceb2:	c4aff06f          	j	1c00c2fc <pos_libc_prf+0x3e>
1c00ceb6:	87d6                	mv	a5,s5
1c00ceb8:	00f69023          	sh	a5,0(a3)
1c00cebc:	c40ff06f          	j	1c00c2fc <pos_libc_prf+0x3e>
1c00cec0:	004a0793          	addi	a5,s4,4
1c00cec4:	ce3e                	sw	a5,28(sp)
1c00cec6:	7761                	lui	a4,0xffff8
1c00cec8:	000a2783          	lw	a5,0(s4)
1c00cecc:	02610593          	addi	a1,sp,38
1c00ced0:	83074713          	xori	a4,a4,-2000
1c00ced4:	4601                	li	a2,0
1c00ced6:	02e11223          	sh	a4,36(sp)
1c00ceda:	86ae                	mv	a3,a1
1c00cedc:	4ea5                	li	t4,9
1c00cede:	f647b733          	p.bclr	a4,a5,27,4
1c00cee2:	01c61513          	slli	a0,a2,0x1c
1c00cee6:	8391                	srli	a5,a5,0x4
1c00cee8:	8fc9                	or	a5,a5,a0
1c00ceea:	8211                	srli	a2,a2,0x4
1c00ceec:	05770e13          	addi	t3,a4,87 # ffff8057 <pulp__FC+0xffff8058>
1c00cef0:	03070893          	addi	a7,a4,48
1c00cef4:	00c7e333          	or	t1,a5,a2
1c00cef8:	00eefa63          	bleu	a4,t4,1c00cf0c <pos_libc_prf+0xc4e>
1c00cefc:	01c68023          	sb	t3,0(a3)
1c00cf00:	00168513          	addi	a0,a3,1
1c00cf04:	00030a63          	beqz	t1,1c00cf18 <pos_libc_prf+0xc5a>
1c00cf08:	86aa                	mv	a3,a0
1c00cf0a:	bfd1                	j	1c00cede <pos_libc_prf+0xc20>
1c00cf0c:	01168023          	sb	a7,0(a3)
1c00cf10:	00168513          	addi	a0,a3,1
1c00cf14:	fe031ae3          	bnez	t1,1c00cf08 <pos_libc_prf+0xc4a>
1c00cf18:	00050023          	sb	zero,0(a0)
1c00cf1c:	8d0d                	sub	a0,a0,a1
1c00cf1e:	00d5fe63          	bleu	a3,a1,1c00cf3a <pos_libc_prf+0xc7c>
1c00cf22:	87b6                	mv	a5,a3
1c00cf24:	86ae                	mv	a3,a1
1c00cf26:	0006c603          	lbu	a2,0(a3)
1c00cf2a:	0007c703          	lbu	a4,0(a5)
1c00cf2e:	fec78fab          	p.sb	a2,-1(a5!)
1c00cf32:	00e680ab          	p.sb	a4,1(a3!)
1c00cf36:	fef6e8e3          	bltu	a3,a5,1c00cf26 <pos_libc_prf+0xc68>
1c00cf3a:	4789                	li	a5,2
1c00cf3c:	00250913          	addi	s2,a0,2
1c00cf40:	c23e                	sw	a5,4(sp)
1c00cf42:	e00ff06f          	j	1c00c542 <pos_libc_prf+0x284>
1c00cf46:	85de                	mv	a1,s7
1c00cf48:	02500513          	li	a0,37
1c00cf4c:	9982                	jalr	s3
1c00cf4e:	cff52363          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00cf52:	85de                	mv	a1,s7
1c00cf54:	856e                	mv	a0,s11
1c00cf56:	9982                	jalr	s3
1c00cf58:	cdf52e63          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00cf5c:	0a89                	addi	s5,s5,2
1c00cf5e:	b9eff06f          	j	1c00c2fc <pos_libc_prf+0x3e>
1c00cf62:	007a0513          	addi	a0,s4,7
1c00cf66:	c4053533          	p.bclr	a0,a0,2,0
1c00cf6a:	411c                	lw	a5,0(a0)
1c00cf6c:	00452303          	lw	t1,4(a0)
1c00cf70:	0521                	addi	a0,a0,8
1c00cf72:	0157d693          	srli	a3,a5,0x15
1c00cf76:	00b31813          	slli	a6,t1,0xb
1c00cf7a:	01435893          	srli	a7,t1,0x14
1c00cf7e:	0106e833          	or	a6,a3,a6
1c00cf82:	ce2a                	sw	a0,28(sp)
1c00cf84:	e8b8b8b3          	p.bclr	a7,a7,20,11
1c00cf88:	00b79593          	slli	a1,a5,0xb
1c00cf8c:	c1f83833          	p.bclr	a6,a6,0,31
1c00cf90:	20034363          	bltz	t1,1c00d196 <pos_libc_prf+0xed8>
1c00cf94:	47b2                	lw	a5,12(sp)
1c00cf96:	1e078463          	beqz	a5,1c00d17e <pos_libc_prf+0xec0>
1c00cf9a:	02b00793          	li	a5,43
1c00cf9e:	02f10223          	sb	a5,36(sp)
1c00cfa2:	02510e13          	addi	t3,sp,37
1c00cfa6:	7ff00793          	li	a5,2047
1c00cfaa:	18f88d63          	beq	a7,a5,1c00d144 <pos_libc_prf+0xe86>
1c00cfae:	04600793          	li	a5,70
1c00cfb2:	0dfdfe93          	andi	t4,s11,223
1c00cfb6:	34fd8363          	beq	s11,a5,1c00d2fc <pos_libc_prf+0x103e>
1c00cfba:	41f8d693          	srai	a3,a7,0x1f
1c00cfbe:	00b8e7b3          	or	a5,a7,a1
1c00cfc2:	0106e6b3          	or	a3,a3,a6
1c00cfc6:	8fd5                	or	a5,a5,a3
1c00cfc8:	8e0781e3          	beqz	a5,1c00c8aa <pos_libc_prf+0x5ec>
1c00cfcc:	02089463          	bnez	a7,1c00cff4 <pos_libc_prf+0xd36>
1c00cfd0:	01f5d793          	srli	a5,a1,0x1f
1c00cfd4:	00181693          	slli	a3,a6,0x1
1c00cfd8:	00d7e833          	or	a6,a5,a3
1c00cfdc:	0586                	slli	a1,a1,0x1
1c00cfde:	5c0843e3          	bltz	a6,1c00dda4 <pos_libc_prf+0x1ae6>
1c00cfe2:	01f5d793          	srli	a5,a1,0x1f
1c00cfe6:	0806                	slli	a6,a6,0x1
1c00cfe8:	0107e833          	or	a6,a5,a6
1c00cfec:	18fd                	addi	a7,a7,-1
1c00cfee:	0586                	slli	a1,a1,0x1
1c00cff0:	fe0859e3          	bgez	a6,1c00cfe2 <pos_libc_prf+0xd24>
1c00cff4:	c0288893          	addi	a7,a7,-1022
1c00cff8:	57f9                	li	a5,-2
1c00cffa:	c1f84833          	p.bset	a6,a6,0,31
1c00cffe:	4301                	li	t1,0
1c00d000:	06f8d863          	ble	a5,a7,1c00d070 <pos_libc_prf+0xdb2>
1c00d004:	33333537          	lui	a0,0x33333
1c00d008:	80000f37          	lui	t5,0x80000
1c00d00c:	4301                	li	t1,0
1c00d00e:	33250513          	addi	a0,a0,818 # 33333332 <__l2_shared_end+0x173221ea>
1c00d012:	ffff4f13          	not	t5,t5
1c00d016:	5ff9                	li	t6,-2
1c00d018:	a011                	j	1c00d01c <pos_libc_prf+0xd5e>
1c00d01a:	88be                	mv	a7,a5
1c00d01c:	01f81693          	slli	a3,a6,0x1f
1c00d020:	0015d713          	srli	a4,a1,0x1
1c00d024:	fc15b7b3          	p.bclr	a5,a1,30,1
1c00d028:	8f55                	or	a4,a4,a3
1c00d02a:	00e785b3          	add	a1,a5,a4
1c00d02e:	00f5b7b3          	sltu	a5,a1,a5
1c00d032:	00185813          	srli	a6,a6,0x1
1c00d036:	983e                	add	a6,a6,a5
1c00d038:	00188793          	addi	a5,a7,1
1c00d03c:	fd056fe3          	bltu	a0,a6,1c00d01a <pos_libc_prf+0xd5c>
1c00d040:	00259693          	slli	a3,a1,0x2
1c00d044:	01e5d613          	srli	a2,a1,0x1e
1c00d048:	00281713          	slli	a4,a6,0x2
1c00d04c:	95b6                	add	a1,a1,a3
1c00d04e:	8f51                	or	a4,a4,a2
1c00d050:	983a                	add	a6,a6,a4
1c00d052:	00d5b6b3          	sltu	a3,a1,a3
1c00d056:	9836                	add	a6,a6,a3
1c00d058:	01f5d713          	srli	a4,a1,0x1f
1c00d05c:	0889                	addi	a7,a7,2
1c00d05e:	137d                	addi	t1,t1,-1
1c00d060:	01e81663          	bne	a6,t5,1c00d06c <pos_libc_prf+0xdae>
1c00d064:	0586                	slli	a1,a1,0x1
1c00d066:	fc174833          	p.bset	a6,a4,30,1
1c00d06a:	88be                	mv	a7,a5
1c00d06c:	fbf8c8e3          	blt	a7,t6,1c00d01c <pos_libc_prf+0xd5e>
1c00d070:	80000f37          	lui	t5,0x80000
1c00d074:	4f95                	li	t6,5
1c00d076:	ffff4f13          	not	t5,t5
1c00d07a:	83105ce3          	blez	a7,1c00c8b2 <pos_libc_prf+0x5f4>
1c00d07e:	00258693          	addi	a3,a1,2
1c00d082:	00b6b5b3          	sltu	a1,a3,a1
1c00d086:	982e                	add	a6,a6,a1
1c00d088:	03f857b3          	divu	a5,a6,t6
1c00d08c:	0036d613          	srli	a2,a3,0x3
1c00d090:	18fd                	addi	a7,a7,-1
1c00d092:	0305                	addi	t1,t1,1
1c00d094:	00279713          	slli	a4,a5,0x2
1c00d098:	973e                	add	a4,a4,a5
1c00d09a:	40e80833          	sub	a6,a6,a4
1c00d09e:	01d81713          	slli	a4,a6,0x1d
1c00d0a2:	8e59                	or	a2,a2,a4
1c00d0a4:	03f65633          	divu	a2,a2,t6
1c00d0a8:	00361593          	slli	a1,a2,0x3
1c00d0ac:	00561713          	slli	a4,a2,0x5
1c00d0b0:	972e                	add	a4,a4,a1
1c00d0b2:	40e68733          	sub	a4,a3,a4
1c00d0b6:	03f75733          	divu	a4,a4,t6
1c00d0ba:	8275                	srli	a2,a2,0x1d
1c00d0bc:	97b2                	add	a5,a5,a2
1c00d0be:	95ba                	add	a1,a1,a4
1c00d0c0:	00e5b733          	sltu	a4,a1,a4
1c00d0c4:	00f70833          	add	a6,a4,a5
1c00d0c8:	01f5d793          	srli	a5,a1,0x1f
1c00d0cc:	0806                	slli	a6,a6,0x1
1c00d0ce:	0107e833          	or	a6,a5,a6
1c00d0d2:	0586                	slli	a1,a1,0x1
1c00d0d4:	18fd                	addi	a7,a7,-1
1c00d0d6:	ff0f79e3          	bleu	a6,t5,1c00d0c8 <pos_libc_prf+0xe0a>
1c00d0da:	b745                	j	1c00d07a <pos_libc_prf+0xdbc>
1c00d0dc:	001d4d83          	lbu	s11,1(s10)
1c00d0e0:	04c00793          	li	a5,76
1c00d0e4:	0d09                	addi	s10,s10,2
1c00d0e6:	ad0ff06f          	j	1c00c3b6 <pos_libc_prf+0xf8>
1c00d0ea:	001d4d83          	lbu	s11,1(s10)
1c00d0ee:	04800793          	li	a5,72
1c00d0f2:	0d09                	addi	s10,s10,2
1c00d0f4:	ac2ff06f          	j	1c00c3b6 <pos_libc_prf+0xf8>
1c00d0f8:	4705                	li	a4,1
1c00d0fa:	02514783          	lbu	a5,37(sp)
1c00d0fe:	c23a                	sw	a4,4(sp)
1c00d100:	b38d                	j	1c00ce62 <pos_libc_prf+0xba4>
1c00d102:	004a0793          	addi	a5,s4,4
1c00d106:	ce3e                	sw	a5,28(sp)
1c00d108:	000a2403          	lw	s0,0(s4)
1c00d10c:	4c81                	li	s9,0
1c00d10e:	e8cff06f          	j	1c00c79a <pos_libc_prf+0x4dc>
1c00d112:	4785                	li	a5,1
1c00d114:	c23e                	sw	a5,4(sp)
1c00d116:	c2cff06f          	j	1c00c542 <pos_libc_prf+0x284>
1c00d11a:	4742                	lw	a4,16(sp)
1c00d11c:	22070063          	beqz	a4,1c00d33c <pos_libc_prf+0x107e>
1c00d120:	02000693          	li	a3,32
1c00d124:	02d10223          	sb	a3,36(sp)
1c00d128:	4c92                	lw	s9,4(sp)
1c00d12a:	02510a13          	addi	s4,sp,37
1c00d12e:	ba6ff06f          	j	1c00c4d4 <pos_libc_prf+0x216>
1c00d132:	04800613          	li	a2,72
1c00d136:	d6c79ce3          	bne	a5,a2,1c00ceae <pos_libc_prf+0xbf0>
1c00d13a:	87d6                	mv	a5,s5
1c00d13c:	00f68023          	sb	a5,0(a3)
1c00d140:	9bcff06f          	j	1c00c2fc <pos_libc_prf+0x3e>
1c00d144:	0105e6b3          	or	a3,a1,a6
1c00d148:	fbfd8613          	addi	a2,s11,-65
1c00d14c:	003e0513          	addi	a0,t3,3
1c00d150:	47e5                	li	a5,25
1c00d152:	16069963          	bnez	a3,1c00d2c4 <pos_libc_prf+0x1006>
1c00d156:	3ec7e1e3          	bltu	a5,a2,1c00dd38 <pos_libc_prf+0x1a7a>
1c00d15a:	6795                	lui	a5,0x5
1c00d15c:	e4978793          	addi	a5,a5,-439 # 4e49 <_l1_preload_size+0x44c9>
1c00d160:	00fe1023          	sh	a5,0(t3)
1c00d164:	04600793          	li	a5,70
1c00d168:	00fe0123          	sb	a5,2(t3)
1c00d16c:	4782                	lw	a5,0(sp)
1c00d16e:	000e01a3          	sb	zero,3(t3)
1c00d172:	4581                	li	a1,0
1c00d174:	8d1d                	sub	a0,a0,a5
1c00d176:	4c81                	li	s9,0
1c00d178:	cc02                	sw	zero,24(sp)
1c00d17a:	4a01                	li	s4,0
1c00d17c:	b1f1                	j	1c00ce48 <pos_libc_prf+0xb8a>
1c00d17e:	4742                	lw	a4,16(sp)
1c00d180:	02410e13          	addi	t3,sp,36
1c00d184:	e20701e3          	beqz	a4,1c00cfa6 <pos_libc_prf+0xce8>
1c00d188:	02000793          	li	a5,32
1c00d18c:	02f10223          	sb	a5,36(sp)
1c00d190:	02510e13          	addi	t3,sp,37
1c00d194:	bd09                	j	1c00cfa6 <pos_libc_prf+0xce8>
1c00d196:	02d00793          	li	a5,45
1c00d19a:	02f10223          	sb	a5,36(sp)
1c00d19e:	02510e13          	addi	t3,sp,37
1c00d1a2:	b511                	j	1c00cfa6 <pos_libc_prf+0xce8>
1c00d1a4:	40f007b3          	neg	a5,a5
1c00d1a8:	00f036b3          	snez	a3,a5
1c00d1ac:	40e00cb3          	neg	s9,a4
1c00d1b0:	02d00613          	li	a2,45
1c00d1b4:	02c10223          	sb	a2,36(sp)
1c00d1b8:	40dc8cb3          	sub	s9,s9,a3
1c00d1bc:	02510a13          	addi	s4,sp,37
1c00d1c0:	b14ff06f          	j	1c00c4d4 <pos_libc_prf+0x216>
1c00d1c4:	57f5                	li	a5,-3
1c00d1c6:	0cf34063          	blt	t1,a5,1c00d286 <pos_libc_prf+0xfc8>
1c00d1ca:	0a64ce63          	blt	s1,t1,1c00d286 <pos_libc_prf+0xfc8>
1c00d1ce:	4792                	lw	a5,4(sp)
1c00d1d0:	406484b3          	sub	s1,s1,t1
1c00d1d4:	4801                	li	a6,0
1c00d1d6:	e789                	bnez	a5,1c00d1e0 <pos_libc_prf+0xf22>
1c00d1d8:	00902833          	sgtz	a6,s1
1c00d1dc:	0ff87813          	andi	a6,a6,255
1c00d1e0:	009307b3          	add	a5,t1,s1
1c00d1e4:	2e07cae3          	bltz	a5,1c00dcd8 <pos_libc_prf+0x1a1a>
1c00d1e8:	4741                	li	a4,16
1c00d1ea:	04e7ceb3          	p.min	t4,a5,a4
1c00d1ee:	1efd                	addi	t4,t4,-1
1c00d1f0:	06600d93          	li	s11,102
1c00d1f4:	f0079863          	bnez	a5,1c00c904 <pos_libc_prf+0x646>
1c00d1f8:	4281                	li	t0,0
1c00d1fa:	080006b7          	lui	a3,0x8000
1c00d1fe:	f70ff06f          	j	1c00c96e <pos_libc_prf+0x6b0>
1c00d202:	86aa                	mv	a3,a0
1c00d204:	01b6812b          	p.sb	s11,2(a3!) # 8000002 <__l1_heap_size+0x7f00986>
1c00d208:	240346e3          	bltz	t1,1c00dc54 <pos_libc_prf+0x1996>
1c00d20c:	02b00793          	li	a5,43
1c00d210:	00f500a3          	sb	a5,1(a0)
1c00d214:	06300793          	li	a5,99
1c00d218:	0067de63          	ble	t1,a5,1c00d234 <pos_libc_prf+0xf76>
1c00d21c:	06400613          	li	a2,100
1c00d220:	02c347b3          	div	a5,t1,a2
1c00d224:	00350693          	addi	a3,a0,3
1c00d228:	02c36333          	rem	t1,t1,a2
1c00d22c:	03078793          	addi	a5,a5,48
1c00d230:	00f50123          	sb	a5,2(a0)
1c00d234:	47a9                	li	a5,10
1c00d236:	8536                	mv	a0,a3
1c00d238:	02f34633          	div	a2,t1,a5
1c00d23c:	02f367b3          	rem	a5,t1,a5
1c00d240:	03060613          	addi	a2,a2,48 # 1a100030 <__l1_end+0xa0ff6ac>
1c00d244:	00c5012b          	p.sb	a2,2(a0!)
1c00d248:	03078793          	addi	a5,a5,48
1c00d24c:	00f680a3          	sb	a5,1(a3)
1c00d250:	bec5                	j	1c00ce40 <pos_libc_prf+0xb82>
1c00d252:	800004b7          	lui	s1,0x80000
1c00d256:	fff4c493          	not	s1,s1
1c00d25a:	c36ff06f          	j	1c00c690 <pos_libc_prf+0x3d2>
1c00d25e:	85e6                	mv	a1,s9
1c00d260:	cc02                	sw	zero,24(sp)
1c00d262:	4a01                	li	s4,0
1c00d264:	bec1                	j	1c00ce34 <pos_libc_prf+0xb76>
1c00d266:	4901                	li	s2,0
1c00d268:	1054                	addi	a3,sp,36
1c00d26a:	d5aff06f          	j	1c00c7c4 <pos_libc_prf+0x506>
1c00d26e:	007a0793          	addi	a5,s4,7
1c00d272:	c407b7b3          	p.bclr	a5,a5,2,0
1c00d276:	00878713          	addi	a4,a5,8
1c00d27a:	ce3a                	sw	a4,28(sp)
1c00d27c:	4380                	lw	s0,0(a5)
1c00d27e:	0047ac83          	lw	s9,4(a5)
1c00d282:	d18ff06f          	j	1c00c79a <pos_libc_prf+0x4dc>
1c00d286:	ffed8613          	addi	a2,s11,-2
1c00d28a:	0ff67d93          	andi	s11,a2,255
1c00d28e:	220486e3          	beqz	s1,1c00dcba <pos_libc_prf+0x19fc>
1c00d292:	4712                	lw	a4,4(sp)
1c00d294:	fff48793          	addi	a5,s1,-1 # 7fffffff <pulp__FC+0x80000000>
1c00d298:	1e071fe3          	bnez	a4,1c00dc96 <pos_libc_prf+0x19d8>
1c00d29c:	00f02833          	sgtz	a6,a5
1c00d2a0:	0ff87813          	andi	a6,a6,255
1c00d2a4:	84be                	mv	s1,a5
1c00d2a6:	e52ff06f          	j	1c00c8f8 <pos_libc_prf+0x63a>
1c00d2aa:	007a0693          	addi	a3,s4,7
1c00d2ae:	c406b6b3          	p.bclr	a3,a3,2,0
1c00d2b2:	00868793          	addi	a5,a3,8
1c00d2b6:	ce3e                	sw	a5,28(sp)
1c00d2b8:	429c                	lw	a5,0(a3)
1c00d2ba:	42d4                	lw	a3,4(a3)
1c00d2bc:	c236                	sw	a3,4(sp)
1c00d2be:	8736                	mv	a4,a3
1c00d2c0:	9fcff06f          	j	1c00c4bc <pos_libc_prf+0x1fe>
1c00d2c4:	24c7efe3          	bltu	a5,a2,1c00dd22 <pos_libc_prf+0x1a64>
1c00d2c8:	6791                	lui	a5,0x4
1c00d2ca:	14e78793          	addi	a5,a5,334 # 414e <_l1_preload_size+0x37ce>
1c00d2ce:	00fe1023          	sh	a5,0(t3)
1c00d2d2:	04e00793          	li	a5,78
1c00d2d6:	00fe0123          	sb	a5,2(t3)
1c00d2da:	bd49                	j	1c00d16c <pos_libc_prf+0xeae>
1c00d2dc:	02e00793          	li	a5,46
1c00d2e0:	00fe00a3          	sb	a5,1(t3)
1c00d2e4:	002e0793          	addi	a5,t3,2
1c00d2e8:	f4904c63          	bgtz	s1,1c00ca40 <pos_libc_prf+0x782>
1c00d2ec:	8ca6                	mv	s9,s1
1c00d2ee:	853e                	mv	a0,a5
1c00d2f0:	be29                	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00d2f2:	4481                	li	s1,0
1c00d2f4:	c202                	sw	zero,4(sp)
1c00d2f6:	4901                	li	s2,0
1c00d2f8:	a64ff06f          	j	1c00c55c <pos_libc_prf+0x29e>
1c00d2fc:	04600e93          	li	t4,70
1c00d300:	06600d93          	li	s11,102
1c00d304:	b95d                	j	1c00cfba <pos_libc_prf+0xcfc>
1c00d306:	105c                	addi	a5,sp,36
1c00d308:	4565                	li	a0,25
1c00d30a:	0017c70b          	p.lbu	a4,1(a5!)
1c00d30e:	f9f70693          	addi	a3,a4,-97
1c00d312:	0ff6f693          	andi	a3,a3,255
1c00d316:	fe070613          	addi	a2,a4,-32
1c00d31a:	00d56e63          	bltu	a0,a3,1c00d336 <pos_libc_prf+0x1078>
1c00d31e:	fec78fa3          	sb	a2,-1(a5)
1c00d322:	0017c70b          	p.lbu	a4,1(a5!)
1c00d326:	f9f70693          	addi	a3,a4,-97
1c00d32a:	0ff6f693          	andi	a3,a3,255
1c00d32e:	fe070613          	addi	a2,a4,-32
1c00d332:	fed576e3          	bleu	a3,a0,1c00d31e <pos_libc_prf+0x1060>
1c00d336:	fb71                	bnez	a4,1c00d30a <pos_libc_prf+0x104c>
1c00d338:	d48ff06f          	j	1c00c880 <pos_libc_prf+0x5c2>
1c00d33c:	4c92                	lw	s9,4(sp)
1c00d33e:	02410a13          	addi	s4,sp,36
1c00d342:	992ff06f          	j	1c00c4d4 <pos_libc_prf+0x216>
1c00d346:	4922                	lw	s2,8(sp)
1c00d348:	8dd2                	mv	s11,s4
1c00d34a:	a029                	j	1c00d354 <pos_libc_prf+0x1096>
1c00d34c:	9982                	jalr	s3
1c00d34e:	1dfd                	addi	s11,s11,-1
1c00d350:	8ff52263          	p.beqimm	a0,-1,1c00c434 <pos_libc_prf+0x176>
1c00d354:	85de                	mv	a1,s7
1c00d356:	03000513          	li	a0,48
1c00d35a:	ffb049e3          	bgtz	s11,1c00d34c <pos_libc_prf+0x108e>
1c00d35e:	41490933          	sub	s2,s2,s4
1c00d362:	a5cff06f          	j	1c00c5be <pos_libc_prf+0x300>
1c00d366:	4712                	lw	a4,4(sp)
1c00d368:	12070fe3          	beqz	a4,1c00dca6 <pos_libc_prf+0x19e8>
1c00d36c:	03000793          	li	a5,48
1c00d370:	02f10223          	sb	a5,36(sp)
1c00d374:	019467b3          	or	a5,s0,s9
1c00d378:	14078ae3          	beqz	a5,1c00dccc <pos_libc_prf+0x1a0e>
1c00d37c:	02510593          	addi	a1,sp,37
1c00d380:	4905                	li	s2,1
1c00d382:	86ae                	mv	a3,a1
1c00d384:	87b6                	mv	a5,a3
1c00d386:	a011                	j	1c00d38a <pos_libc_prf+0x10cc>
1c00d388:	87b2                	mv	a5,a2
1c00d38a:	01dc9613          	slli	a2,s9,0x1d
1c00d38e:	f8343733          	p.bclr	a4,s0,28,3
1c00d392:	800d                	srli	s0,s0,0x3
1c00d394:	8c51                	or	s0,s0,a2
1c00d396:	03070713          	addi	a4,a4,48
1c00d39a:	003cdc93          	srli	s9,s9,0x3
1c00d39e:	00e78023          	sb	a4,0(a5)
1c00d3a2:	01946733          	or	a4,s0,s9
1c00d3a6:	00178613          	addi	a2,a5,1
1c00d3aa:	ff79                	bnez	a4,1c00d388 <pos_libc_prf+0x10ca>
1c00d3ac:	00060023          	sb	zero,0(a2)
1c00d3b0:	8e0d                	sub	a2,a2,a1
1c00d3b2:	00f6fc63          	bleu	a5,a3,1c00d3ca <pos_libc_prf+0x110c>
1c00d3b6:	0006c583          	lbu	a1,0(a3)
1c00d3ba:	0007c703          	lbu	a4,0(a5)
1c00d3be:	feb78fab          	p.sb	a1,-1(a5!)
1c00d3c2:	00e680ab          	p.sb	a4,1(a3!)
1c00d3c6:	fef6e8e3          	bltu	a3,a5,1c00d3b6 <pos_libc_prf+0x10f8>
1c00d3ca:	9932                	add	s2,s2,a2
1c00d3cc:	c202                	sw	zero,4(sp)
1c00d3ce:	974ff06f          	j	1c00c542 <pos_libc_prf+0x284>
1c00d3d2:	106057e3          	blez	t1,1c00dce0 <pos_libc_prf+0x1a22>
1c00d3d6:	00279e93          	slli	t4,a5,0x2
1c00d3da:	01e7df93          	srli	t6,a5,0x1e
1c00d3de:	00269f13          	slli	t5,a3,0x2
1c00d3e2:	00fe85b3          	add	a1,t4,a5
1c00d3e6:	01efef33          	or	t5,t6,t5
1c00d3ea:	9f36                	add	t5,t5,a3
1c00d3ec:	01d5b6b3          	sltu	a3,a1,t4
1c00d3f0:	96fa                	add	a3,a3,t5
1c00d3f2:	01f5d793          	srli	a5,a1,0x1f
1c00d3f6:	0686                	slli	a3,a3,0x1
1c00d3f8:	8edd                	or	a3,a3,a5
1c00d3fa:	01c6d793          	srli	a5,a3,0x1c
1c00d3fe:	03078793          	addi	a5,a5,48
1c00d402:	10000eb7          	lui	t4,0x10000
1c00d406:	1efd                	addi	t4,t4,-1
1c00d408:	00fe0023          	sb	a5,0(t3)
1c00d40c:	01d6f6b3          	and	a3,a3,t4
1c00d410:	00159793          	slli	a5,a1,0x1
1c00d414:	16132ee3          	p.beqimm	t1,1,1c00dd90 <pos_libc_prf+0x1ad2>
1c00d418:	01e7df93          	srli	t6,a5,0x1e
1c00d41c:	00269f13          	slli	t5,a3,0x2
1c00d420:	058e                	slli	a1,a1,0x3
1c00d422:	00f58533          	add	a0,a1,a5
1c00d426:	01efef33          	or	t5,t6,t5
1c00d42a:	9f36                	add	t5,t5,a3
1c00d42c:	00b536b3          	sltu	a3,a0,a1
1c00d430:	96fa                	add	a3,a3,t5
1c00d432:	01f55793          	srli	a5,a0,0x1f
1c00d436:	0686                	slli	a3,a3,0x1
1c00d438:	8edd                	or	a3,a3,a5
1c00d43a:	01c6d793          	srli	a5,a3,0x1c
1c00d43e:	03078793          	addi	a5,a5,48
1c00d442:	00fe00a3          	sb	a5,1(t3)
1c00d446:	01d6f6b3          	and	a3,a3,t4
1c00d44a:	00151793          	slli	a5,a0,0x1
1c00d44e:	162323e3          	p.beqimm	t1,2,1c00ddb4 <pos_libc_prf+0x1af6>
1c00d452:	01e7df93          	srli	t6,a5,0x1e
1c00d456:	00269f13          	slli	t5,a3,0x2
1c00d45a:	050e                	slli	a0,a0,0x3
1c00d45c:	00f505b3          	add	a1,a0,a5
1c00d460:	01efef33          	or	t5,t6,t5
1c00d464:	9f36                	add	t5,t5,a3
1c00d466:	00a5b6b3          	sltu	a3,a1,a0
1c00d46a:	96fa                	add	a3,a3,t5
1c00d46c:	01f5d793          	srli	a5,a1,0x1f
1c00d470:	0686                	slli	a3,a3,0x1
1c00d472:	8edd                	or	a3,a3,a5
1c00d474:	01c6d793          	srli	a5,a3,0x1c
1c00d478:	03078793          	addi	a5,a5,48
1c00d47c:	00fe0123          	sb	a5,2(t3)
1c00d480:	ffd30a13          	addi	s4,t1,-3
1c00d484:	00159793          	slli	a5,a1,0x1
1c00d488:	01d6f6b3          	and	a3,a3,t4
1c00d48c:	134050e3          	blez	s4,1c00ddac <pos_libc_prf+0x1aee>
1c00d490:	01e7df93          	srli	t6,a5,0x1e
1c00d494:	00269f13          	slli	t5,a3,0x2
1c00d498:	058e                	slli	a1,a1,0x3
1c00d49a:	00f58533          	add	a0,a1,a5
1c00d49e:	01efef33          	or	t5,t6,t5
1c00d4a2:	9f36                	add	t5,t5,a3
1c00d4a4:	00b536b3          	sltu	a3,a0,a1
1c00d4a8:	96fa                	add	a3,a3,t5
1c00d4aa:	01f55793          	srli	a5,a0,0x1f
1c00d4ae:	0686                	slli	a3,a3,0x1
1c00d4b0:	8edd                	or	a3,a3,a5
1c00d4b2:	01c6d793          	srli	a5,a3,0x1c
1c00d4b6:	03078793          	addi	a5,a5,48
1c00d4ba:	00fe01a3          	sb	a5,3(t3)
1c00d4be:	ffc30a13          	addi	s4,t1,-4
1c00d4c2:	00151793          	slli	a5,a0,0x1
1c00d4c6:	01d6f6b3          	and	a3,a3,t4
1c00d4ca:	134056e3          	blez	s4,1c00ddf6 <pos_libc_prf+0x1b38>
1c00d4ce:	01e7df93          	srli	t6,a5,0x1e
1c00d4d2:	00269f13          	slli	t5,a3,0x2
1c00d4d6:	050e                	slli	a0,a0,0x3
1c00d4d8:	00f505b3          	add	a1,a0,a5
1c00d4dc:	01efef33          	or	t5,t6,t5
1c00d4e0:	9f36                	add	t5,t5,a3
1c00d4e2:	00a5b6b3          	sltu	a3,a1,a0
1c00d4e6:	96fa                	add	a3,a3,t5
1c00d4e8:	01f5d793          	srli	a5,a1,0x1f
1c00d4ec:	0686                	slli	a3,a3,0x1
1c00d4ee:	8edd                	or	a3,a3,a5
1c00d4f0:	01c6d793          	srli	a5,a3,0x1c
1c00d4f4:	03078793          	addi	a5,a5,48
1c00d4f8:	00fe0223          	sb	a5,4(t3)
1c00d4fc:	ffb30a13          	addi	s4,t1,-5
1c00d500:	00159793          	slli	a5,a1,0x1
1c00d504:	01d6f6b3          	and	a3,a3,t4
1c00d508:	0f4053e3          	blez	s4,1c00ddee <pos_libc_prf+0x1b30>
1c00d50c:	01e7df93          	srli	t6,a5,0x1e
1c00d510:	00269f13          	slli	t5,a3,0x2
1c00d514:	058e                	slli	a1,a1,0x3
1c00d516:	00f58533          	add	a0,a1,a5
1c00d51a:	01efef33          	or	t5,t6,t5
1c00d51e:	9f36                	add	t5,t5,a3
1c00d520:	00b536b3          	sltu	a3,a0,a1
1c00d524:	96fa                	add	a3,a3,t5
1c00d526:	01f55793          	srli	a5,a0,0x1f
1c00d52a:	0686                	slli	a3,a3,0x1
1c00d52c:	8edd                	or	a3,a3,a5
1c00d52e:	01c6d793          	srli	a5,a3,0x1c
1c00d532:	03078793          	addi	a5,a5,48
1c00d536:	00fe02a3          	sb	a5,5(t3)
1c00d53a:	ffa30a13          	addi	s4,t1,-6
1c00d53e:	00151793          	slli	a5,a0,0x1
1c00d542:	01d6f6b3          	and	a3,a3,t4
1c00d546:	0b4050e3          	blez	s4,1c00dde6 <pos_libc_prf+0x1b28>
1c00d54a:	01e7df93          	srli	t6,a5,0x1e
1c00d54e:	00269f13          	slli	t5,a3,0x2
1c00d552:	050e                	slli	a0,a0,0x3
1c00d554:	00f505b3          	add	a1,a0,a5
1c00d558:	01efef33          	or	t5,t6,t5
1c00d55c:	9f36                	add	t5,t5,a3
1c00d55e:	00a5b6b3          	sltu	a3,a1,a0
1c00d562:	96fa                	add	a3,a3,t5
1c00d564:	01f5d793          	srli	a5,a1,0x1f
1c00d568:	0686                	slli	a3,a3,0x1
1c00d56a:	8edd                	or	a3,a3,a5
1c00d56c:	01c6d793          	srli	a5,a3,0x1c
1c00d570:	03078793          	addi	a5,a5,48
1c00d574:	00fe0323          	sb	a5,6(t3)
1c00d578:	ff930a13          	addi	s4,t1,-7
1c00d57c:	00159793          	slli	a5,a1,0x1
1c00d580:	01d6f6b3          	and	a3,a3,t4
1c00d584:	05405de3          	blez	s4,1c00ddde <pos_libc_prf+0x1b20>
1c00d588:	058e                	slli	a1,a1,0x3
1c00d58a:	01e7df13          	srli	t5,a5,0x1e
1c00d58e:	00269513          	slli	a0,a3,0x2
1c00d592:	97ae                	add	a5,a5,a1
1c00d594:	00af6533          	or	a0,t5,a0
1c00d598:	9536                	add	a0,a0,a3
1c00d59a:	00b7b6b3          	sltu	a3,a5,a1
1c00d59e:	96aa                	add	a3,a3,a0
1c00d5a0:	01f7d593          	srli	a1,a5,0x1f
1c00d5a4:	0686                	slli	a3,a3,0x1
1c00d5a6:	8ecd                	or	a3,a3,a1
1c00d5a8:	01c6d593          	srli	a1,a3,0x1c
1c00d5ac:	03058593          	addi	a1,a1,48
1c00d5b0:	0786                	slli	a5,a5,0x1
1c00d5b2:	00be03a3          	sb	a1,7(t3)
1c00d5b6:	ff830a13          	addi	s4,t1,-8
1c00d5ba:	85be                	mv	a1,a5
1c00d5bc:	01d6f6b3          	and	a3,a3,t4
1c00d5c0:	01405be3          	blez	s4,1c00ddd6 <pos_libc_prf+0x1b18>
1c00d5c4:	078a                	slli	a5,a5,0x2
1c00d5c6:	01e5de93          	srli	t4,a1,0x1e
1c00d5ca:	00269513          	slli	a0,a3,0x2
1c00d5ce:	95be                	add	a1,a1,a5
1c00d5d0:	00aee533          	or	a0,t4,a0
1c00d5d4:	9536                	add	a0,a0,a3
1c00d5d6:	00f5b6b3          	sltu	a3,a1,a5
1c00d5da:	96aa                	add	a3,a3,a0
1c00d5dc:	01f5d793          	srli	a5,a1,0x1f
1c00d5e0:	0686                	slli	a3,a3,0x1
1c00d5e2:	8edd                	or	a3,a3,a5
1c00d5e4:	01c6d793          	srli	a5,a3,0x1c
1c00d5e8:	03078793          	addi	a5,a5,48
1c00d5ec:	10000eb7          	lui	t4,0x10000
1c00d5f0:	1efd                	addi	t4,t4,-1
1c00d5f2:	00fe0423          	sb	a5,8(t3)
1c00d5f6:	ff730a13          	addi	s4,t1,-9
1c00d5fa:	00159793          	slli	a5,a1,0x1
1c00d5fe:	01d6f6b3          	and	a3,a3,t4
1c00d602:	7d405663          	blez	s4,1c00ddce <pos_libc_prf+0x1b10>
1c00d606:	01e7df93          	srli	t6,a5,0x1e
1c00d60a:	00269f13          	slli	t5,a3,0x2
1c00d60e:	058e                	slli	a1,a1,0x3
1c00d610:	00f58533          	add	a0,a1,a5
1c00d614:	01efef33          	or	t5,t6,t5
1c00d618:	9f36                	add	t5,t5,a3
1c00d61a:	00b536b3          	sltu	a3,a0,a1
1c00d61e:	96fa                	add	a3,a3,t5
1c00d620:	01f55793          	srli	a5,a0,0x1f
1c00d624:	0686                	slli	a3,a3,0x1
1c00d626:	8edd                	or	a3,a3,a5
1c00d628:	01c6d793          	srli	a5,a3,0x1c
1c00d62c:	03078793          	addi	a5,a5,48
1c00d630:	00fe04a3          	sb	a5,9(t3)
1c00d634:	ff630a13          	addi	s4,t1,-10
1c00d638:	00151793          	slli	a5,a0,0x1
1c00d63c:	01d6f6b3          	and	a3,a3,t4
1c00d640:	79405363          	blez	s4,1c00ddc6 <pos_libc_prf+0x1b08>
1c00d644:	01e7df93          	srli	t6,a5,0x1e
1c00d648:	00269f13          	slli	t5,a3,0x2
1c00d64c:	050e                	slli	a0,a0,0x3
1c00d64e:	00f505b3          	add	a1,a0,a5
1c00d652:	01efef33          	or	t5,t6,t5
1c00d656:	9f36                	add	t5,t5,a3
1c00d658:	00a5b6b3          	sltu	a3,a1,a0
1c00d65c:	96fa                	add	a3,a3,t5
1c00d65e:	01f5d793          	srli	a5,a1,0x1f
1c00d662:	0686                	slli	a3,a3,0x1
1c00d664:	8edd                	or	a3,a3,a5
1c00d666:	01c6d793          	srli	a5,a3,0x1c
1c00d66a:	03078793          	addi	a5,a5,48
1c00d66e:	00fe0523          	sb	a5,10(t3)
1c00d672:	ff530a13          	addi	s4,t1,-11
1c00d676:	00159793          	slli	a5,a1,0x1
1c00d67a:	01d6f6b3          	and	a3,a3,t4
1c00d67e:	75405063          	blez	s4,1c00ddbe <pos_libc_prf+0x1b00>
1c00d682:	01e7df93          	srli	t6,a5,0x1e
1c00d686:	00269f13          	slli	t5,a3,0x2
1c00d68a:	058e                	slli	a1,a1,0x3
1c00d68c:	00f58533          	add	a0,a1,a5
1c00d690:	01efef33          	or	t5,t6,t5
1c00d694:	9f36                	add	t5,t5,a3
1c00d696:	00b536b3          	sltu	a3,a0,a1
1c00d69a:	96fa                	add	a3,a3,t5
1c00d69c:	01f55793          	srli	a5,a0,0x1f
1c00d6a0:	0686                	slli	a3,a3,0x1
1c00d6a2:	8edd                	or	a3,a3,a5
1c00d6a4:	01c6d793          	srli	a5,a3,0x1c
1c00d6a8:	03078793          	addi	a5,a5,48
1c00d6ac:	00fe05a3          	sb	a5,11(t3)
1c00d6b0:	ff430a13          	addi	s4,t1,-12
1c00d6b4:	00151793          	slli	a5,a0,0x1
1c00d6b8:	01d6f6b3          	and	a3,a3,t4
1c00d6bc:	6f405163          	blez	s4,1c00dd9e <pos_libc_prf+0x1ae0>
1c00d6c0:	050e                	slli	a0,a0,0x3
1c00d6c2:	01e7df13          	srli	t5,a5,0x1e
1c00d6c6:	00269893          	slli	a7,a3,0x2
1c00d6ca:	00f505b3          	add	a1,a0,a5
1c00d6ce:	011f68b3          	or	a7,t5,a7
1c00d6d2:	98b6                	add	a7,a7,a3
1c00d6d4:	00a5b6b3          	sltu	a3,a1,a0
1c00d6d8:	96c6                	add	a3,a3,a7
1c00d6da:	01f5d793          	srli	a5,a1,0x1f
1c00d6de:	0686                	slli	a3,a3,0x1
1c00d6e0:	8edd                	or	a3,a3,a5
1c00d6e2:	01c6d793          	srli	a5,a3,0x1c
1c00d6e6:	03078793          	addi	a5,a5,48
1c00d6ea:	00fe0623          	sb	a5,12(t3)
1c00d6ee:	ff330a13          	addi	s4,t1,-13
1c00d6f2:	00159793          	slli	a5,a1,0x1
1c00d6f6:	01d6f6b3          	and	a3,a3,t4
1c00d6fa:	00de0513          	addi	a0,t3,13
1c00d6fe:	488d                	li	a7,3
1c00d700:	0d405163          	blez	s4,1c00d7c2 <pos_libc_prf+0x1504>
1c00d704:	01e7df13          	srli	t5,a5,0x1e
1c00d708:	00269893          	slli	a7,a3,0x2
1c00d70c:	058e                	slli	a1,a1,0x3
1c00d70e:	00f58533          	add	a0,a1,a5
1c00d712:	011f68b3          	or	a7,t5,a7
1c00d716:	98b6                	add	a7,a7,a3
1c00d718:	00b536b3          	sltu	a3,a0,a1
1c00d71c:	96c6                	add	a3,a3,a7
1c00d71e:	01f55793          	srli	a5,a0,0x1f
1c00d722:	0686                	slli	a3,a3,0x1
1c00d724:	8edd                	or	a3,a3,a5
1c00d726:	01c6d793          	srli	a5,a3,0x1c
1c00d72a:	03078793          	addi	a5,a5,48
1c00d72e:	00fe06a3          	sb	a5,13(t3)
1c00d732:	ff230a13          	addi	s4,t1,-14
1c00d736:	00151793          	slli	a5,a0,0x1
1c00d73a:	01d6f6b3          	and	a3,a3,t4
1c00d73e:	65405c63          	blez	s4,1c00dd96 <pos_libc_prf+0x1ad8>
1c00d742:	050e                	slli	a0,a0,0x3
1c00d744:	01e7df13          	srli	t5,a5,0x1e
1c00d748:	00269893          	slli	a7,a3,0x2
1c00d74c:	00f505b3          	add	a1,a0,a5
1c00d750:	011f68b3          	or	a7,t5,a7
1c00d754:	98b6                	add	a7,a7,a3
1c00d756:	00a5b6b3          	sltu	a3,a1,a0
1c00d75a:	96c6                	add	a3,a3,a7
1c00d75c:	01f5d793          	srli	a5,a1,0x1f
1c00d760:	0686                	slli	a3,a3,0x1
1c00d762:	8edd                	or	a3,a3,a5
1c00d764:	01c6d793          	srli	a5,a3,0x1c
1c00d768:	03078793          	addi	a5,a5,48
1c00d76c:	00fe0723          	sb	a5,14(t3)
1c00d770:	ff130a13          	addi	s4,t1,-15
1c00d774:	00159793          	slli	a5,a1,0x1
1c00d778:	01d6f6b3          	and	a3,a3,t4
1c00d77c:	00fe0513          	addi	a0,t3,15
1c00d780:	4885                	li	a7,1
1c00d782:	05405063          	blez	s4,1c00d7c2 <pos_libc_prf+0x1504>
1c00d786:	01e7d893          	srli	a7,a5,0x1e
1c00d78a:	058e                	slli	a1,a1,0x3
1c00d78c:	00269513          	slli	a0,a3,0x2
1c00d790:	97ae                	add	a5,a5,a1
1c00d792:	00a8e533          	or	a0,a7,a0
1c00d796:	9536                	add	a0,a0,a3
1c00d798:	00b7b6b3          	sltu	a3,a5,a1
1c00d79c:	96aa                	add	a3,a3,a0
1c00d79e:	01f7d593          	srli	a1,a5,0x1f
1c00d7a2:	0686                	slli	a3,a3,0x1
1c00d7a4:	8ecd                	or	a3,a3,a1
1c00d7a6:	01c6d593          	srli	a1,a3,0x1c
1c00d7aa:	03058593          	addi	a1,a1,48
1c00d7ae:	010e0513          	addi	a0,t3,16
1c00d7b2:	0786                	slli	a5,a5,0x1
1c00d7b4:	01d6f6b3          	and	a3,a3,t4
1c00d7b8:	00be07a3          	sb	a1,15(t3)
1c00d7bc:	ff030a13          	addi	s4,t1,-16
1c00d7c0:	4881                	li	a7,0
1c00d7c2:	4712                	lw	a4,4(sp)
1c00d7c4:	4e070663          	beqz	a4,1c00dcb0 <pos_libc_prf+0x19f2>
1c00d7c8:	02e00313          	li	t1,46
1c00d7cc:	00650023          	sb	t1,0(a0)
1c00d7d0:	8e26                	mv	t3,s1
1c00d7d2:	cc02                	sw	zero,24(sp)
1c00d7d4:	00150593          	addi	a1,a0,1
1c00d7d8:	4301                	li	t1,0
1c00d7da:	63c05263          	blez	t3,1c00ddfe <pos_libc_prf+0x1b40>
1c00d7de:	62088063          	beqz	a7,1c00ddfe <pos_libc_prf+0x1b40>
1c00d7e2:	00279513          	slli	a0,a5,0x2
1c00d7e6:	01e7df13          	srli	t5,a5,0x1e
1c00d7ea:	00269e93          	slli	t4,a3,0x2
1c00d7ee:	97aa                	add	a5,a5,a0
1c00d7f0:	01df6eb3          	or	t4,t5,t4
1c00d7f4:	96f6                	add	a3,a3,t4
1c00d7f6:	00a7bf33          	sltu	t5,a5,a0
1c00d7fa:	9f36                	add	t5,t5,a3
1c00d7fc:	0f06                	slli	t5,t5,0x1
1c00d7fe:	01f7d693          	srli	a3,a5,0x1f
1c00d802:	01e6ef33          	or	t5,a3,t5
1c00d806:	01cf5693          	srli	a3,t5,0x1c
1c00d80a:	03068693          	addi	a3,a3,48
1c00d80e:	00d58023          	sb	a3,0(a1)
1c00d812:	fffe0493          	addi	s1,t3,-1
1c00d816:	00158513          	addi	a0,a1,1
1c00d81a:	00179693          	slli	a3,a5,0x1
1c00d81e:	42048263          	beqz	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d822:	4218a063          	p.beqimm	a7,1,1c00dc42 <pos_libc_prf+0x1984>
1c00d826:	10000eb7          	lui	t4,0x10000
1c00d82a:	1efd                	addi	t4,t4,-1
1c00d82c:	01df7f33          	and	t5,t5,t4
1c00d830:	078e                	slli	a5,a5,0x3
1c00d832:	01e6df93          	srli	t6,a3,0x1e
1c00d836:	002f1513          	slli	a0,t5,0x2
1c00d83a:	96be                	add	a3,a3,a5
1c00d83c:	00afe533          	or	a0,t6,a0
1c00d840:	9f2a                	add	t5,t5,a0
1c00d842:	00f6b7b3          	sltu	a5,a3,a5
1c00d846:	97fa                	add	a5,a5,t5
1c00d848:	01f6d513          	srli	a0,a3,0x1f
1c00d84c:	0786                	slli	a5,a5,0x1
1c00d84e:	8fc9                	or	a5,a5,a0
1c00d850:	01c7d513          	srli	a0,a5,0x1c
1c00d854:	03050513          	addi	a0,a0,48
1c00d858:	00a580a3          	sb	a0,1(a1)
1c00d85c:	ffee0493          	addi	s1,t3,-2
1c00d860:	00258513          	addi	a0,a1,2
1c00d864:	00169f13          	slli	t5,a3,0x1
1c00d868:	01d7f7b3          	and	a5,a5,t4
1c00d86c:	3c048b63          	beqz	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d870:	3c28a963          	p.beqimm	a7,2,1c00dc42 <pos_libc_prf+0x1984>
1c00d874:	01ef5f93          	srli	t6,t5,0x1e
1c00d878:	068e                	slli	a3,a3,0x3
1c00d87a:	00279513          	slli	a0,a5,0x2
1c00d87e:	9f36                	add	t5,t5,a3
1c00d880:	00afe533          	or	a0,t6,a0
1c00d884:	00df36b3          	sltu	a3,t5,a3
1c00d888:	97aa                	add	a5,a5,a0
1c00d88a:	97b6                	add	a5,a5,a3
1c00d88c:	0786                	slli	a5,a5,0x1
1c00d88e:	01ff5693          	srli	a3,t5,0x1f
1c00d892:	8fd5                	or	a5,a5,a3
1c00d894:	01c7d693          	srli	a3,a5,0x1c
1c00d898:	03068693          	addi	a3,a3,48
1c00d89c:	00d58123          	sb	a3,2(a1)
1c00d8a0:	ffde0493          	addi	s1,t3,-3
1c00d8a4:	00358513          	addi	a0,a1,3
1c00d8a8:	001f1693          	slli	a3,t5,0x1
1c00d8ac:	01d7ffb3          	and	t6,a5,t4
1c00d8b0:	38905963          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d8b4:	3838a763          	p.beqimm	a7,3,1c00dc42 <pos_libc_prf+0x1984>
1c00d8b8:	003f1513          	slli	a0,t5,0x3
1c00d8bc:	01e6d793          	srli	a5,a3,0x1e
1c00d8c0:	002f9f13          	slli	t5,t6,0x2
1c00d8c4:	96aa                	add	a3,a3,a0
1c00d8c6:	01e7ef33          	or	t5,a5,t5
1c00d8ca:	9f7e                	add	t5,t5,t6
1c00d8cc:	00a6b7b3          	sltu	a5,a3,a0
1c00d8d0:	97fa                	add	a5,a5,t5
1c00d8d2:	01f6d513          	srli	a0,a3,0x1f
1c00d8d6:	0786                	slli	a5,a5,0x1
1c00d8d8:	8fc9                	or	a5,a5,a0
1c00d8da:	01c7d513          	srli	a0,a5,0x1c
1c00d8de:	03050513          	addi	a0,a0,48
1c00d8e2:	00a581a3          	sb	a0,3(a1)
1c00d8e6:	ffce0493          	addi	s1,t3,-4
1c00d8ea:	00458513          	addi	a0,a1,4
1c00d8ee:	00169f13          	slli	t5,a3,0x1
1c00d8f2:	01d7feb3          	and	t4,a5,t4
1c00d8f6:	34905663          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d8fa:	3448a463          	p.beqimm	a7,4,1c00dc42 <pos_libc_prf+0x1984>
1c00d8fe:	068e                	slli	a3,a3,0x3
1c00d900:	01ef5793          	srli	a5,t5,0x1e
1c00d904:	002e9513          	slli	a0,t4,0x2
1c00d908:	9f36                	add	t5,t5,a3
1c00d90a:	8d5d                	or	a0,a0,a5
1c00d90c:	9eaa                	add	t4,t4,a0
1c00d90e:	00df37b3          	sltu	a5,t5,a3
1c00d912:	97f6                	add	a5,a5,t4
1c00d914:	01ff5693          	srli	a3,t5,0x1f
1c00d918:	0786                	slli	a5,a5,0x1
1c00d91a:	8fd5                	or	a5,a5,a3
1c00d91c:	01c7d693          	srli	a3,a5,0x1c
1c00d920:	03068693          	addi	a3,a3,48
1c00d924:	00d58223          	sb	a3,4(a1)
1c00d928:	ffbe0493          	addi	s1,t3,-5
1c00d92c:	00558513          	addi	a0,a1,5
1c00d930:	001f1693          	slli	a3,t5,0x1
1c00d934:	30905763          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d938:	3058a563          	p.beqimm	a7,5,1c00dc42 <pos_libc_prf+0x1984>
1c00d93c:	10000eb7          	lui	t4,0x10000
1c00d940:	1efd                	addi	t4,t4,-1
1c00d942:	01d7f7b3          	and	a5,a5,t4
1c00d946:	01e6df13          	srli	t5,a3,0x1e
1c00d94a:	00269513          	slli	a0,a3,0x2
1c00d94e:	00279f93          	slli	t6,a5,0x2
1c00d952:	96aa                	add	a3,a3,a0
1c00d954:	01ff6fb3          	or	t6,t5,t6
1c00d958:	00a6b533          	sltu	a0,a3,a0
1c00d95c:	97fe                	add	a5,a5,t6
1c00d95e:	97aa                	add	a5,a5,a0
1c00d960:	00179f13          	slli	t5,a5,0x1
1c00d964:	01f6d513          	srli	a0,a3,0x1f
1c00d968:	01e56f33          	or	t5,a0,t5
1c00d96c:	01cf5793          	srli	a5,t5,0x1c
1c00d970:	03078793          	addi	a5,a5,48
1c00d974:	00f582a3          	sb	a5,5(a1)
1c00d978:	ffae0493          	addi	s1,t3,-6
1c00d97c:	00658513          	addi	a0,a1,6
1c00d980:	00169793          	slli	a5,a3,0x1
1c00d984:	01df7f33          	and	t5,t5,t4
1c00d988:	2a905d63          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d98c:	2a68ab63          	p.beqimm	a7,6,1c00dc42 <pos_libc_prf+0x1984>
1c00d990:	068e                	slli	a3,a3,0x3
1c00d992:	01e7df93          	srli	t6,a5,0x1e
1c00d996:	002f1513          	slli	a0,t5,0x2
1c00d99a:	97b6                	add	a5,a5,a3
1c00d99c:	00afe533          	or	a0,t6,a0
1c00d9a0:	9f2a                	add	t5,t5,a0
1c00d9a2:	00d7b6b3          	sltu	a3,a5,a3
1c00d9a6:	96fa                	add	a3,a3,t5
1c00d9a8:	01f7d513          	srli	a0,a5,0x1f
1c00d9ac:	00169f13          	slli	t5,a3,0x1
1c00d9b0:	01e56f33          	or	t5,a0,t5
1c00d9b4:	01cf5693          	srli	a3,t5,0x1c
1c00d9b8:	03068693          	addi	a3,a3,48
1c00d9bc:	00d58323          	sb	a3,6(a1)
1c00d9c0:	ff9e0493          	addi	s1,t3,-7
1c00d9c4:	00758513          	addi	a0,a1,7
1c00d9c8:	00179693          	slli	a3,a5,0x1
1c00d9cc:	01df7f33          	and	t5,t5,t4
1c00d9d0:	26905963          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00d9d4:	2678a763          	p.beqimm	a7,7,1c00dc42 <pos_libc_prf+0x1984>
1c00d9d8:	078e                	slli	a5,a5,0x3
1c00d9da:	01e6df93          	srli	t6,a3,0x1e
1c00d9de:	002f1513          	slli	a0,t5,0x2
1c00d9e2:	96be                	add	a3,a3,a5
1c00d9e4:	00afe533          	or	a0,t6,a0
1c00d9e8:	9f2a                	add	t5,t5,a0
1c00d9ea:	00f6b7b3          	sltu	a5,a3,a5
1c00d9ee:	97fa                	add	a5,a5,t5
1c00d9f0:	01f6d513          	srli	a0,a3,0x1f
1c00d9f4:	00179f13          	slli	t5,a5,0x1
1c00d9f8:	01e56f33          	or	t5,a0,t5
1c00d9fc:	01cf5793          	srli	a5,t5,0x1c
1c00da00:	03078793          	addi	a5,a5,48
1c00da04:	00f583a3          	sb	a5,7(a1)
1c00da08:	ff8e0493          	addi	s1,t3,-8
1c00da0c:	00858513          	addi	a0,a1,8
1c00da10:	00169793          	slli	a5,a3,0x1
1c00da14:	01df7f33          	and	t5,t5,t4
1c00da18:	22905563          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00da1c:	2288a363          	p.beqimm	a7,8,1c00dc42 <pos_libc_prf+0x1984>
1c00da20:	01e7df93          	srli	t6,a5,0x1e
1c00da24:	068e                	slli	a3,a3,0x3
1c00da26:	002f1513          	slli	a0,t5,0x2
1c00da2a:	97b6                	add	a5,a5,a3
1c00da2c:	00afe533          	or	a0,t6,a0
1c00da30:	9f2a                	add	t5,t5,a0
1c00da32:	00d7b6b3          	sltu	a3,a5,a3
1c00da36:	96fa                	add	a3,a3,t5
1c00da38:	01f7d513          	srli	a0,a5,0x1f
1c00da3c:	0686                	slli	a3,a3,0x1
1c00da3e:	8ec9                	or	a3,a3,a0
1c00da40:	01c6d513          	srli	a0,a3,0x1c
1c00da44:	03050513          	addi	a0,a0,48
1c00da48:	00a58423          	sb	a0,8(a1)
1c00da4c:	ff7e0493          	addi	s1,t3,-9
1c00da50:	00958513          	addi	a0,a1,9
1c00da54:	00179f93          	slli	t6,a5,0x1
1c00da58:	01d6ff33          	and	t5,a3,t4
1c00da5c:	1e905363          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00da60:	1e98a163          	p.beqimm	a7,9,1c00dc42 <pos_libc_prf+0x1984>
1c00da64:	078e                	slli	a5,a5,0x3
1c00da66:	01efd293          	srli	t0,t6,0x1e
1c00da6a:	002f1513          	slli	a0,t5,0x2
1c00da6e:	01f786b3          	add	a3,a5,t6
1c00da72:	00a2e533          	or	a0,t0,a0
1c00da76:	00f6b7b3          	sltu	a5,a3,a5
1c00da7a:	957a                	add	a0,a0,t5
1c00da7c:	97aa                	add	a5,a5,a0
1c00da7e:	0786                	slli	a5,a5,0x1
1c00da80:	01f6d513          	srli	a0,a3,0x1f
1c00da84:	8fc9                	or	a5,a5,a0
1c00da86:	01c7d513          	srli	a0,a5,0x1c
1c00da8a:	03050513          	addi	a0,a0,48
1c00da8e:	00a584a3          	sb	a0,9(a1)
1c00da92:	ff6e0493          	addi	s1,t3,-10
1c00da96:	00a58513          	addi	a0,a1,10
1c00da9a:	0686                	slli	a3,a3,0x1
1c00da9c:	01d7f7b3          	and	a5,a5,t4
1c00daa0:	1a905163          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00daa4:	18a8af63          	p.beqimm	a7,10,1c00dc42 <pos_libc_prf+0x1984>
1c00daa8:	01e6df13          	srli	t5,a3,0x1e
1c00daac:	00269513          	slli	a0,a3,0x2
1c00dab0:	00279e93          	slli	t4,a5,0x2
1c00dab4:	96aa                	add	a3,a3,a0
1c00dab6:	01df6eb3          	or	t4,t5,t4
1c00daba:	00a6b533          	sltu	a0,a3,a0
1c00dabe:	97f6                	add	a5,a5,t4
1c00dac0:	97aa                	add	a5,a5,a0
1c00dac2:	00179f13          	slli	t5,a5,0x1
1c00dac6:	01f6d513          	srli	a0,a3,0x1f
1c00daca:	01e56f33          	or	t5,a0,t5
1c00dace:	01cf5793          	srli	a5,t5,0x1c
1c00dad2:	03078793          	addi	a5,a5,48
1c00dad6:	10000eb7          	lui	t4,0x10000
1c00dada:	1efd                	addi	t4,t4,-1
1c00dadc:	00f58523          	sb	a5,10(a1)
1c00dae0:	ff5e0493          	addi	s1,t3,-11
1c00dae4:	00b58513          	addi	a0,a1,11
1c00dae8:	00169793          	slli	a5,a3,0x1
1c00daec:	01df7f33          	and	t5,t5,t4
1c00daf0:	14905963          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00daf4:	14b8a763          	p.beqimm	a7,11,1c00dc42 <pos_libc_prf+0x1984>
1c00daf8:	068e                	slli	a3,a3,0x3
1c00dafa:	01e7df93          	srli	t6,a5,0x1e
1c00dafe:	002f1513          	slli	a0,t5,0x2
1c00db02:	97b6                	add	a5,a5,a3
1c00db04:	00afe533          	or	a0,t6,a0
1c00db08:	9f2a                	add	t5,t5,a0
1c00db0a:	00d7b6b3          	sltu	a3,a5,a3
1c00db0e:	96fa                	add	a3,a3,t5
1c00db10:	01f7d513          	srli	a0,a5,0x1f
1c00db14:	0686                	slli	a3,a3,0x1
1c00db16:	8ec9                	or	a3,a3,a0
1c00db18:	01c6d513          	srli	a0,a3,0x1c
1c00db1c:	03050513          	addi	a0,a0,48
1c00db20:	00a585a3          	sb	a0,11(a1)
1c00db24:	ff4e0493          	addi	s1,t3,-12
1c00db28:	00c58513          	addi	a0,a1,12
1c00db2c:	00179f13          	slli	t5,a5,0x1
1c00db30:	10905963          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00db34:	10c8a763          	p.beqimm	a7,12,1c00dc42 <pos_libc_prf+0x1984>
1c00db38:	01d6f6b3          	and	a3,a3,t4
1c00db3c:	01ef5f93          	srli	t6,t5,0x1e
1c00db40:	078e                	slli	a5,a5,0x3
1c00db42:	00269513          	slli	a0,a3,0x2
1c00db46:	9f3e                	add	t5,t5,a5
1c00db48:	00afe533          	or	a0,t6,a0
1c00db4c:	96aa                	add	a3,a3,a0
1c00db4e:	00ff37b3          	sltu	a5,t5,a5
1c00db52:	97b6                	add	a5,a5,a3
1c00db54:	01ff5513          	srli	a0,t5,0x1f
1c00db58:	00179693          	slli	a3,a5,0x1
1c00db5c:	8ec9                	or	a3,a3,a0
1c00db5e:	01c6d793          	srli	a5,a3,0x1c
1c00db62:	03078793          	addi	a5,a5,48
1c00db66:	00f58623          	sb	a5,12(a1)
1c00db6a:	ff3e0493          	addi	s1,t3,-13
1c00db6e:	00d58513          	addi	a0,a1,13
1c00db72:	001f1793          	slli	a5,t5,0x1
1c00db76:	01d6ffb3          	and	t6,a3,t4
1c00db7a:	0c905463          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00db7e:	0cd8a263          	p.beqimm	a7,13,1c00dc42 <pos_libc_prf+0x1984>
1c00db82:	003f1693          	slli	a3,t5,0x3
1c00db86:	002f9513          	slli	a0,t6,0x2
1c00db8a:	01e7df13          	srli	t5,a5,0x1e
1c00db8e:	00af6533          	or	a0,t5,a0
1c00db92:	97b6                	add	a5,a5,a3
1c00db94:	957e                	add	a0,a0,t6
1c00db96:	00d7b6b3          	sltu	a3,a5,a3
1c00db9a:	96aa                	add	a3,a3,a0
1c00db9c:	0686                	slli	a3,a3,0x1
1c00db9e:	01f7d513          	srli	a0,a5,0x1f
1c00dba2:	8ec9                	or	a3,a3,a0
1c00dba4:	01c6d513          	srli	a0,a3,0x1c
1c00dba8:	03050513          	addi	a0,a0,48
1c00dbac:	00a586a3          	sb	a0,13(a1)
1c00dbb0:	ff2e0493          	addi	s1,t3,-14
1c00dbb4:	00e58513          	addi	a0,a1,14
1c00dbb8:	00179f93          	slli	t6,a5,0x1
1c00dbbc:	01d6ff33          	and	t5,a3,t4
1c00dbc0:	08905163          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00dbc4:	06e8af63          	p.beqimm	a7,14,1c00dc42 <pos_libc_prf+0x1984>
1c00dbc8:	078e                	slli	a5,a5,0x3
1c00dbca:	002f1513          	slli	a0,t5,0x2
1c00dbce:	01efd293          	srli	t0,t6,0x1e
1c00dbd2:	01f786b3          	add	a3,a5,t6
1c00dbd6:	00a2e533          	or	a0,t0,a0
1c00dbda:	957a                	add	a0,a0,t5
1c00dbdc:	00f6b7b3          	sltu	a5,a3,a5
1c00dbe0:	97aa                	add	a5,a5,a0
1c00dbe2:	00179f13          	slli	t5,a5,0x1
1c00dbe6:	01f6d513          	srli	a0,a3,0x1f
1c00dbea:	01e56f33          	or	t5,a0,t5
1c00dbee:	01cf5793          	srli	a5,t5,0x1c
1c00dbf2:	03078793          	addi	a5,a5,48
1c00dbf6:	00f58723          	sb	a5,14(a1)
1c00dbfa:	ff1e0493          	addi	s1,t3,-15
1c00dbfe:	00f58513          	addi	a0,a1,15
1c00dc02:	00169793          	slli	a5,a3,0x1
1c00dc06:	01df7eb3          	and	t4,t5,t4
1c00dc0a:	02905c63          	blez	s1,1c00dc42 <pos_libc_prf+0x1984>
1c00dc0e:	46c1                	li	a3,16
1c00dc10:	02d89963          	bne	a7,a3,1c00dc42 <pos_libc_prf+0x1984>
1c00dc14:	00279513          	slli	a0,a5,0x2
1c00dc18:	01e7d893          	srli	a7,a5,0x1e
1c00dc1c:	002e9693          	slli	a3,t4,0x2
1c00dc20:	97aa                	add	a5,a5,a0
1c00dc22:	00d8e6b3          	or	a3,a7,a3
1c00dc26:	00a7b7b3          	sltu	a5,a5,a0
1c00dc2a:	9eb6                	add	t4,t4,a3
1c00dc2c:	97f6                	add	a5,a5,t4
1c00dc2e:	0786                	slli	a5,a5,0x1
1c00dc30:	83f1                	srli	a5,a5,0x1c
1c00dc32:	03078793          	addi	a5,a5,48
1c00dc36:	01058513          	addi	a0,a1,16
1c00dc3a:	00f587a3          	sb	a5,15(a1)
1c00dc3e:	ff0e0493          	addi	s1,t3,-16
1c00dc42:	47e2                	lw	a5,24(sp)
1c00dc44:	014785b3          	add	a1,a5,s4
1c00dc48:	9c081663          	bnez	a6,1c00ce14 <pos_libc_prf+0xb56>
1c00dc4c:	95a6                	add	a1,a1,s1
1c00dc4e:	8ca6                	mv	s9,s1
1c00dc50:	9f0ff06f          	j	1c00ce40 <pos_libc_prf+0xb82>
1c00dc54:	02d00793          	li	a5,45
1c00dc58:	40600333          	neg	t1,t1
1c00dc5c:	00f500a3          	sb	a5,1(a0)
1c00dc60:	db4ff06f          	j	1c00d214 <pos_libc_prf+0xf56>
1c00dc64:	41fad793          	srai	a5,s5,0x1f
1c00dc68:	0156a023          	sw	s5,0(a3)
1c00dc6c:	c2dc                	sw	a5,4(a3)
1c00dc6e:	e8efe06f          	j	1c00c2fc <pos_libc_prf+0x3e>
1c00dc72:	4a62                	lw	s4,24(sp)
1c00dc74:	a039                	j	1c00dc82 <pos_libc_prf+0x19c4>
1c00dc76:	9982                	jalr	s3
1c00dc78:	1a7d                	addi	s4,s4,-1
1c00dc7a:	01f53463          	p.bneimm	a0,-1,1c00dc82 <pos_libc_prf+0x19c4>
1c00dc7e:	fb6fe06f          	j	1c00c434 <pos_libc_prf+0x176>
1c00dc82:	85de                	mv	a1,s7
1c00dc84:	03000513          	li	a0,48
1c00dc88:	ff4047e3          	bgtz	s4,1c00dc76 <pos_libc_prf+0x19b8>
1c00dc8c:	4762                	lw	a4,24(sp)
1c00dc8e:	40e90933          	sub	s2,s2,a4
1c00dc92:	933fe06f          	j	1c00c5c4 <pos_libc_prf+0x306>
1c00dc96:	4741                	li	a4,16
1c00dc98:	04e4ceb3          	p.min	t4,s1,a4
1c00dc9c:	1efd                	addi	t4,t4,-1
1c00dc9e:	84be                	mv	s1,a5
1c00dca0:	4801                	li	a6,0
1c00dca2:	c63fe06f          	j	1c00c904 <pos_libc_prf+0x646>
1c00dca6:	104c                	addi	a1,sp,36
1c00dca8:	4901                	li	s2,0
1c00dcaa:	86ae                	mv	a3,a1
1c00dcac:	ed8ff06f          	j	1c00d384 <pos_libc_prf+0x10c6>
1c00dcb0:	0c904463          	bgtz	s1,1c00dd78 <pos_libc_prf+0x1aba>
1c00dcb4:	4301                	li	t1,0
1c00dcb6:	cc02                	sw	zero,24(sp)
1c00dcb8:	b769                	j	1c00dc42 <pos_libc_prf+0x1984>
1c00dcba:	4712                	lw	a4,4(sp)
1c00dcbc:	4801                	li	a6,0
1c00dcbe:	e319                	bnez	a4,1c00dcc4 <pos_libc_prf+0x1a06>
1c00dcc0:	c39fe06f          	j	1c00c8f8 <pos_libc_prf+0x63a>
1c00dcc4:	4e81                	li	t4,0
1c00dcc6:	4801                	li	a6,0
1c00dcc8:	c3dfe06f          	j	1c00c904 <pos_libc_prf+0x646>
1c00dccc:	020102a3          	sb	zero,37(sp)
1c00dcd0:	c202                	sw	zero,4(sp)
1c00dcd2:	4905                	li	s2,1
1c00dcd4:	86ffe06f          	j	1c00c542 <pos_libc_prf+0x284>
1c00dcd8:	06600d93          	li	s11,102
1c00dcdc:	d1cff06f          	j	1c00d1f8 <pos_libc_prf+0xf3a>
1c00dce0:	4712                	lw	a4,4(sp)
1c00dce2:	03000593          	li	a1,48
1c00dce6:	00be0023          	sb	a1,0(t3)
1c00dcea:	e335                	bnez	a4,1c00dd4e <pos_libc_prf+0x1a90>
1c00dcec:	06905b63          	blez	s1,1c00dd62 <pos_libc_prf+0x1aa4>
1c00dcf0:	02e00593          	li	a1,46
1c00dcf4:	00be00a3          	sb	a1,1(t3)
1c00dcf8:	002e0513          	addi	a0,t3,2
1c00dcfc:	48c1                	li	a7,16
1c00dcfe:	08030363          	beqz	t1,1c00dd84 <pos_libc_prf+0x1ac6>
1c00dd02:	40600733          	neg	a4,t1
1c00dd06:	04974733          	p.min	a4,a4,s1
1c00dd0a:	cc3a                	sw	a4,24(sp)
1c00dd0c:	40e48e33          	sub	t3,s1,a4
1c00dd10:	85aa                	mv	a1,a0
1c00dd12:	4a01                	li	s4,0
1c00dd14:	48c1                	li	a7,16
1c00dd16:	adc044e3          	bgtz	t3,1c00d7de <pos_libc_prf+0x1520>
1c00dd1a:	a0d5                	j	1c00ddfe <pos_libc_prf+0x1b40>
1c00dd1c:	4a72                	lw	s4,28(sp)
1c00dd1e:	ddefe06f          	j	1c00c2fc <pos_libc_prf+0x3e>
1c00dd22:	6799                	lui	a5,0x6
1c00dd24:	16e78793          	addi	a5,a5,366 # 616e <_l1_preload_size+0x57ee>
1c00dd28:	00fe1023          	sh	a5,0(t3)
1c00dd2c:	06e00793          	li	a5,110
1c00dd30:	00fe0123          	sb	a5,2(t3)
1c00dd34:	c38ff06f          	j	1c00d16c <pos_libc_prf+0xeae>
1c00dd38:	679d                	lui	a5,0x7
1c00dd3a:	e6978793          	addi	a5,a5,-407 # 6e69 <_l1_preload_size+0x64e9>
1c00dd3e:	00fe1023          	sh	a5,0(t3)
1c00dd42:	06600793          	li	a5,102
1c00dd46:	00fe0123          	sb	a5,2(t3)
1c00dd4a:	c22ff06f          	j	1c00d16c <pos_libc_prf+0xeae>
1c00dd4e:	02e00593          	li	a1,46
1c00dd52:	00be00a3          	sb	a1,1(t3)
1c00dd56:	002e0513          	addi	a0,t3,2
1c00dd5a:	00030763          	beqz	t1,1c00dd68 <pos_libc_prf+0x1aaa>
1c00dd5e:	fa9042e3          	bgtz	s1,1c00dd02 <pos_libc_prf+0x1a44>
1c00dd62:	4a01                	li	s4,0
1c00dd64:	cc02                	sw	zero,24(sp)
1c00dd66:	bdf1                	j	1c00dc42 <pos_libc_prf+0x1984>
1c00dd68:	fe905de3          	blez	s1,1c00dd62 <pos_libc_prf+0x1aa4>
1c00dd6c:	85aa                	mv	a1,a0
1c00dd6e:	4a01                	li	s4,0
1c00dd70:	cc02                	sw	zero,24(sp)
1c00dd72:	8e26                	mv	t3,s1
1c00dd74:	48c1                	li	a7,16
1c00dd76:	b4b5                	j	1c00d7e2 <pos_libc_prf+0x1524>
1c00dd78:	02e00593          	li	a1,46
1c00dd7c:	00b50023          	sb	a1,0(a0)
1c00dd80:	8352                	mv	t1,s4
1c00dd82:	0505                	addi	a0,a0,1
1c00dd84:	8a1a                	mv	s4,t1
1c00dd86:	85aa                	mv	a1,a0
1c00dd88:	8e26                	mv	t3,s1
1c00dd8a:	cc02                	sw	zero,24(sp)
1c00dd8c:	4301                	li	t1,0
1c00dd8e:	bc81                	j	1c00d7de <pos_libc_prf+0x1520>
1c00dd90:	4a01                	li	s4,0
1c00dd92:	48bd                	li	a7,15
1c00dd94:	b43d                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00dd96:	00ee0513          	addi	a0,t3,14
1c00dd9a:	4889                	li	a7,2
1c00dd9c:	b41d                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00dd9e:	00ce0513          	addi	a0,t3,12
1c00dda2:	b405                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00dda4:	c0200893          	li	a7,-1022
1c00dda8:	a5cff06f          	j	1c00d004 <pos_libc_prf+0xd46>
1c00ddac:	003e0513          	addi	a0,t3,3
1c00ddb0:	48b5                	li	a7,13
1c00ddb2:	bc01                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddb4:	002e0513          	addi	a0,t3,2
1c00ddb8:	4a01                	li	s4,0
1c00ddba:	48b9                	li	a7,14
1c00ddbc:	b419                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddbe:	00be0513          	addi	a0,t3,11
1c00ddc2:	4895                	li	a7,5
1c00ddc4:	bafd                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddc6:	00ae0513          	addi	a0,t3,10
1c00ddca:	4899                	li	a7,6
1c00ddcc:	badd                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddce:	009e0513          	addi	a0,t3,9
1c00ddd2:	489d                	li	a7,7
1c00ddd4:	b2fd                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddd6:	008e0513          	addi	a0,t3,8
1c00ddda:	48a1                	li	a7,8
1c00dddc:	b2dd                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddde:	007e0513          	addi	a0,t3,7
1c00dde2:	48a5                	li	a7,9
1c00dde4:	baf9                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00dde6:	006e0513          	addi	a0,t3,6
1c00ddea:	48a9                	li	a7,10
1c00ddec:	bad9                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddee:	005e0513          	addi	a0,t3,5
1c00ddf2:	48ad                	li	a7,11
1c00ddf4:	b2f9                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddf6:	004e0513          	addi	a0,t3,4
1c00ddfa:	48b1                	li	a7,12
1c00ddfc:	b2d9                	j	1c00d7c2 <pos_libc_prf+0x1504>
1c00ddfe:	84f2                	mv	s1,t3
1c00de00:	852e                	mv	a0,a1
1c00de02:	b581                	j	1c00dc42 <pos_libc_prf+0x1984>
1c00de04:	41990933          	sub	s2,s2,s9
1c00de08:	8edfe06f          	j	1c00c6f4 <pos_libc_prf+0x436>
1c00de0c:	00d78513          	addi	a0,a5,13
1c00de10:	ffbfe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de14:	00c78513          	addi	a0,a5,12
1c00de18:	ff3fe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de1c:	00b78513          	addi	a0,a5,11
1c00de20:	febfe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de24:	00978513          	addi	a0,a5,9
1c00de28:	fe3fe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de2c:	00878513          	addi	a0,a5,8
1c00de30:	fdbfe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de34:	00778513          	addi	a0,a5,7
1c00de38:	fd3fe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de3c:	00678513          	addi	a0,a5,6
1c00de40:	fcbfe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de44:	00578513          	addi	a0,a5,5
1c00de48:	fc3fe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de4c:	00478513          	addi	a0,a5,4
1c00de50:	fbbfe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de54:	00378513          	addi	a0,a5,3
1c00de58:	fb3fe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de5c:	00278513          	addi	a0,a5,2
1c00de60:	fabfe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>
1c00de64:	00178513          	addi	a0,a5,1
1c00de68:	4c81                	li	s9,0
1c00de6a:	fa1fe06f          	j	1c00ce0a <pos_libc_prf+0xb4c>

1c00de6e <pos_init_start>:
1c00de6e:	1141                	addi	sp,sp,-16
1c00de70:	c422                	sw	s0,8(sp)
1c00de72:	1c001437          	lui	s0,0x1c001
1c00de76:	c606                	sw	ra,12(sp)
1c00de78:	84c40413          	addi	s0,s0,-1972 # 1c00084c <ctor_list>
1c00de7c:	2445                	jal	1c00e11c <pos_irq_init>
1c00de7e:	aeefe0ef          	jal	ra,1c00c16c <pos_soc_init>
1c00de82:	24f1                	jal	1c00e14e <pos_soc_event_init>
1c00de84:	2a6d                	jal	1c00e03e <pos_allocs_init>
1c00de86:	284d                	jal	1c00df38 <pos_sched_init>
1c00de88:	405c                	lw	a5,4(s0)
1c00de8a:	c791                	beqz	a5,1c00de96 <pos_init_start+0x28>
1c00de8c:	0421                	addi	s0,s0,8
1c00de8e:	9782                	jalr	a5
1c00de90:	0044278b          	p.lw	a5,4(s0!)
1c00de94:	ffed                	bnez	a5,1c00de8e <pos_init_start+0x20>
1c00de96:	bf2fe0ef          	jal	ra,1c00c288 <pos_io_start>
1c00de9a:	300467f3          	csrrsi	a5,mstatus,8
1c00de9e:	40b2                	lw	ra,12(sp)
1c00dea0:	4422                	lw	s0,8(sp)
1c00dea2:	0141                	addi	sp,sp,16
1c00dea4:	8082                	ret

1c00dea6 <pos_init_stop>:
1c00dea6:	1141                	addi	sp,sp,-16
1c00dea8:	c422                	sw	s0,8(sp)
1c00deaa:	1c001437          	lui	s0,0x1c001
1c00deae:	c606                	sw	ra,12(sp)
1c00deb0:	86040413          	addi	s0,s0,-1952 # 1c000860 <dtor_list>
1c00deb4:	bd8fe0ef          	jal	ra,1c00c28c <pos_io_stop>
1c00deb8:	405c                	lw	a5,4(s0)
1c00deba:	c791                	beqz	a5,1c00dec6 <pos_init_stop+0x20>
1c00debc:	0421                	addi	s0,s0,8
1c00debe:	9782                	jalr	a5
1c00dec0:	0044278b          	p.lw	a5,4(s0!)
1c00dec4:	ffed                	bnez	a5,1c00debe <pos_init_stop+0x18>
1c00dec6:	40b2                	lw	ra,12(sp)
1c00dec8:	4422                	lw	s0,8(sp)
1c00deca:	0141                	addi	sp,sp,16
1c00decc:	8082                	ret

1c00dece <pos_cbsys_add>:
1c00dece:	00259793          	slli	a5,a1,0x2
1c00ded2:	1c0025b7          	lui	a1,0x1c002
1c00ded6:	97058593          	addi	a1,a1,-1680 # 1c001970 <stack>
1c00deda:	95be                	add	a1,a1,a5
1c00dedc:	419c                	lw	a5,0(a1)
1c00dede:	c110                	sw	a2,0(a0)
1c00dee0:	c154                	sw	a3,4(a0)
1c00dee2:	c51c                	sw	a5,8(a0)
1c00dee4:	c188                	sw	a0,0(a1)
1c00dee6:	8082                	ret

1c00dee8 <pi_open_from_conf>:
1c00dee8:	c14c                	sw	a1,4(a0)
1c00deea:	8082                	ret

1c00deec <pos_task_handle_blocking>:
1c00deec:	4785                	li	a5,1
1c00deee:	00f50a23          	sb	a5,20(a0)
1c00def2:	8082                	ret

1c00def4 <pos_task_handle>:
1c00def4:	1141                	addi	sp,sp,-16
1c00def6:	c422                	sw	s0,8(sp)
1c00def8:	02000413          	li	s0,32
1c00defc:	401c                	lw	a5,0(s0)
1c00defe:	c606                	sw	ra,12(sp)
1c00df00:	c385                	beqz	a5,1c00df20 <pos_task_handle+0x2c>
1c00df02:	4394                	lw	a3,0(a5)
1c00df04:	43d8                	lw	a4,4(a5)
1c00df06:	4788                	lw	a0,8(a5)
1c00df08:	c014                	sw	a3,0(s0)
1c00df0a:	300467f3          	csrrsi	a5,mstatus,8
1c00df0e:	9702                	jalr	a4
1c00df10:	300477f3          	csrrci	a5,mstatus,8
1c00df14:	401c                	lw	a5,0(s0)
1c00df16:	f7f5                	bnez	a5,1c00df02 <pos_task_handle+0xe>
1c00df18:	40b2                	lw	ra,12(sp)
1c00df1a:	4422                	lw	s0,8(sp)
1c00df1c:	0141                	addi	sp,sp,16
1c00df1e:	8082                	ret
1c00df20:	10500073          	wfi
1c00df24:	300467f3          	csrrsi	a5,mstatus,8
1c00df28:	300477f3          	csrrci	a5,mstatus,8
1c00df2c:	401c                	lw	a5,0(s0)
1c00df2e:	fbf1                	bnez	a5,1c00df02 <pos_task_handle+0xe>
1c00df30:	40b2                	lw	ra,12(sp)
1c00df32:	4422                	lw	s0,8(sp)
1c00df34:	0141                	addi	sp,sp,16
1c00df36:	8082                	ret

1c00df38 <pos_sched_init>:
1c00df38:	02002023          	sw	zero,32(zero) # 20 <pos_sched_first>
1c00df3c:	8082                	ret

1c00df3e <pos_alloc_init>:
1c00df3e:	00758793          	addi	a5,a1,7
1c00df42:	c407b7b3          	p.bclr	a5,a5,2,0
1c00df46:	40b785b3          	sub	a1,a5,a1
1c00df4a:	c11c                	sw	a5,0(a0)
1c00df4c:	8e0d                	sub	a2,a2,a1
1c00df4e:	00c05763          	blez	a2,1c00df5c <pos_alloc_init+0x1e>
1c00df52:	c4063633          	p.bclr	a2,a2,2,0
1c00df56:	c390                	sw	a2,0(a5)
1c00df58:	0007a223          	sw	zero,4(a5)
1c00df5c:	8082                	ret

1c00df5e <pos_alloc>:
1c00df5e:	4110                	lw	a2,0(a0)
1c00df60:	059d                	addi	a1,a1,7
1c00df62:	c405b5b3          	p.bclr	a1,a1,2,0
1c00df66:	ca31                	beqz	a2,1c00dfba <pos_alloc+0x5c>
1c00df68:	4218                	lw	a4,0(a2)
1c00df6a:	425c                	lw	a5,4(a2)
1c00df6c:	02b75963          	ble	a1,a4,1c00df9e <pos_alloc+0x40>
1c00df70:	cb81                	beqz	a5,1c00df80 <pos_alloc+0x22>
1c00df72:	4398                	lw	a4,0(a5)
1c00df74:	43d4                	lw	a3,4(a5)
1c00df76:	00b75763          	ble	a1,a4,1c00df84 <pos_alloc+0x26>
1c00df7a:	863e                	mv	a2,a5
1c00df7c:	87b6                	mv	a5,a3
1c00df7e:	fbf5                	bnez	a5,1c00df72 <pos_alloc+0x14>
1c00df80:	853e                	mv	a0,a5
1c00df82:	8082                	ret
1c00df84:	00b70a63          	beq	a4,a1,1c00df98 <pos_alloc+0x3a>
1c00df88:	00b78533          	add	a0,a5,a1
1c00df8c:	8f0d                	sub	a4,a4,a1
1c00df8e:	c118                	sw	a4,0(a0)
1c00df90:	c154                	sw	a3,4(a0)
1c00df92:	c248                	sw	a0,4(a2)
1c00df94:	853e                	mv	a0,a5
1c00df96:	8082                	ret
1c00df98:	c254                	sw	a3,4(a2)
1c00df9a:	853e                	mv	a0,a5
1c00df9c:	8082                	ret
1c00df9e:	00e58b63          	beq	a1,a4,1c00dfb4 <pos_alloc+0x56>
1c00dfa2:	00b606b3          	add	a3,a2,a1
1c00dfa6:	8f0d                	sub	a4,a4,a1
1c00dfa8:	c2dc                	sw	a5,4(a3)
1c00dfaa:	c298                	sw	a4,0(a3)
1c00dfac:	87b2                	mv	a5,a2
1c00dfae:	c114                	sw	a3,0(a0)
1c00dfb0:	853e                	mv	a0,a5
1c00dfb2:	8082                	ret
1c00dfb4:	c11c                	sw	a5,0(a0)
1c00dfb6:	87b2                	mv	a5,a2
1c00dfb8:	b7e1                	j	1c00df80 <pos_alloc+0x22>
1c00dfba:	4781                	li	a5,0
1c00dfbc:	b7d1                	j	1c00df80 <pos_alloc+0x22>

1c00dfbe <pos_free>:
1c00dfbe:	4118                	lw	a4,0(a0)
1c00dfc0:	061d                	addi	a2,a2,7
1c00dfc2:	c4063633          	p.bclr	a2,a2,2,0
1c00dfc6:	c729                	beqz	a4,1c00e010 <pos_free+0x52>
1c00dfc8:	00b76663          	bltu	a4,a1,1c00dfd4 <pos_free+0x16>
1c00dfcc:	a0a1                	j	1c00e014 <pos_free+0x56>
1c00dfce:	00b7f563          	bleu	a1,a5,1c00dfd8 <pos_free+0x1a>
1c00dfd2:	873e                	mv	a4,a5
1c00dfd4:	435c                	lw	a5,4(a4)
1c00dfd6:	ffe5                	bnez	a5,1c00dfce <pos_free+0x10>
1c00dfd8:	00c586b3          	add	a3,a1,a2
1c00dfdc:	02d78463          	beq	a5,a3,1c00e004 <pos_free+0x46>
1c00dfe0:	c190                	sw	a2,0(a1)
1c00dfe2:	c1dc                	sw	a5,4(a1)
1c00dfe4:	cf11                	beqz	a4,1c00e000 <pos_free+0x42>
1c00dfe6:	431c                	lw	a5,0(a4)
1c00dfe8:	00f706b3          	add	a3,a4,a5
1c00dfec:	00d58463          	beq	a1,a3,1c00dff4 <pos_free+0x36>
1c00dff0:	c34c                	sw	a1,4(a4)
1c00dff2:	8082                	ret
1c00dff4:	4194                	lw	a3,0(a1)
1c00dff6:	41d0                	lw	a2,4(a1)
1c00dff8:	97b6                	add	a5,a5,a3
1c00dffa:	c31c                	sw	a5,0(a4)
1c00dffc:	c350                	sw	a2,4(a4)
1c00dffe:	8082                	ret
1c00e000:	c10c                	sw	a1,0(a0)
1c00e002:	8082                	ret
1c00e004:	4394                	lw	a3,0(a5)
1c00e006:	43dc                	lw	a5,4(a5)
1c00e008:	9636                	add	a2,a2,a3
1c00e00a:	c190                	sw	a2,0(a1)
1c00e00c:	c1dc                	sw	a5,4(a1)
1c00e00e:	bfd9                	j	1c00dfe4 <pos_free+0x26>
1c00e010:	4781                	li	a5,0
1c00e012:	b7d9                	j	1c00dfd8 <pos_free+0x1a>
1c00e014:	87ba                	mv	a5,a4
1c00e016:	4701                	li	a4,0
1c00e018:	b7c1                	j	1c00dfd8 <pos_free+0x1a>

1c00e01a <pos_alloc_init_l1>:
1c00e01a:	1c0027b7          	lui	a5,0x1c002
1c00e01e:	9d47a703          	lw	a4,-1580(a5) # 1c0019d4 <pos_alloc_l1>
1c00e022:	100017b7          	lui	a5,0x10001
1c00e026:	01651593          	slli	a1,a0,0x16
1c00e02a:	000ff637          	lui	a2,0xff
1c00e02e:	050a                	slli	a0,a0,0x2
1c00e030:	98478793          	addi	a5,a5,-1660 # 10000984 <__l1_end>
1c00e034:	67c60613          	addi	a2,a2,1660 # ff67c <__l1_heap_size>
1c00e038:	95be                	add	a1,a1,a5
1c00e03a:	953a                	add	a0,a0,a4
1c00e03c:	b709                	j	1c00df3e <pos_alloc_init>

1c00e03e <pos_allocs_init>:
1c00e03e:	1141                	addi	sp,sp,-16
1c00e040:	1c0025b7          	lui	a1,0x1c002
1c00e044:	c606                	sw	ra,12(sp)
1c00e046:	c422                	sw	s0,8(sp)
1c00e048:	a0058793          	addi	a5,a1,-1536 # 1c001a00 <__l2_priv0_end>
1c00e04c:	1c008637          	lui	a2,0x1c008
1c00e050:	04c7c363          	blt	a5,a2,1c00e096 <pos_allocs_init+0x58>
1c00e054:	4581                	li	a1,0
1c00e056:	4601                	li	a2,0
1c00e058:	1c002437          	lui	s0,0x1c002
1c00e05c:	9d840513          	addi	a0,s0,-1576 # 1c0019d8 <pos_alloc_l2>
1c00e060:	3df9                	jal	1c00df3e <pos_alloc_init>
1c00e062:	1c0115b7          	lui	a1,0x1c011
1c00e066:	14858793          	addi	a5,a1,328 # 1c011148 <__l2_shared_end>
1c00e06a:	1c080637          	lui	a2,0x1c080
1c00e06e:	1c002537          	lui	a0,0x1c002
1c00e072:	8e1d                	sub	a2,a2,a5
1c00e074:	14858593          	addi	a1,a1,328
1c00e078:	9dc50513          	addi	a0,a0,-1572 # 1c0019dc <pos_alloc_l2+0x4>
1c00e07c:	35c9                	jal	1c00df3e <pos_alloc_init>
1c00e07e:	9d840513          	addi	a0,s0,-1576
1c00e082:	4591                	li	a1,4
1c00e084:	3de9                	jal	1c00df5e <pos_alloc>
1c00e086:	40b2                	lw	ra,12(sp)
1c00e088:	4422                	lw	s0,8(sp)
1c00e08a:	1c0027b7          	lui	a5,0x1c002
1c00e08e:	9ca7aa23          	sw	a0,-1580(a5) # 1c0019d4 <pos_alloc_l1>
1c00e092:	0141                	addi	sp,sp,16
1c00e094:	8082                	ret
1c00e096:	8e1d                	sub	a2,a2,a5
1c00e098:	a0058593          	addi	a1,a1,-1536
1c00e09c:	bf75                	j	1c00e058 <pos_allocs_init+0x1a>

1c00e09e <pi_cl_l1_malloc>:
1c00e09e:	4781                	li	a5,0
1c00e0a0:	c509                	beqz	a0,1c00e0aa <pi_cl_l1_malloc+0xc>
1c00e0a2:	451c                	lw	a5,8(a0)
1c00e0a4:	01c7c783          	lbu	a5,28(a5)
1c00e0a8:	078a                	slli	a5,a5,0x2
1c00e0aa:	1c002737          	lui	a4,0x1c002
1c00e0ae:	9d472503          	lw	a0,-1580(a4) # 1c0019d4 <pos_alloc_l1>
1c00e0b2:	953e                	add	a0,a0,a5
1c00e0b4:	b56d                	j	1c00df5e <pos_alloc>

1c00e0b6 <pi_cl_l1_free>:
1c00e0b6:	4781                	li	a5,0
1c00e0b8:	c509                	beqz	a0,1c00e0c2 <pi_cl_l1_free+0xc>
1c00e0ba:	451c                	lw	a5,8(a0)
1c00e0bc:	01c7c783          	lbu	a5,28(a5)
1c00e0c0:	078a                	slli	a5,a5,0x2
1c00e0c2:	1c002737          	lui	a4,0x1c002
1c00e0c6:	9d472503          	lw	a0,-1580(a4) # 1c0019d4 <pos_alloc_l1>
1c00e0ca:	953e                	add	a0,a0,a5
1c00e0cc:	bdcd                	j	1c00dfbe <pos_free>

1c00e0ce <pos_irq_set_handler>:
1c00e0ce:	f14027f3          	csrr	a5,mhartid
1c00e0d2:	477d                	li	a4,31
1c00e0d4:	ca5797b3          	p.extractu	a5,a5,5,5
1c00e0d8:	02e78d63          	beq	a5,a4,1c00e112 <pos_irq_set_handler+0x44>
1c00e0dc:	002007b7          	lui	a5,0x200
1c00e0e0:	43b8                	lw	a4,64(a5)
1c00e0e2:	050a                	slli	a0,a0,0x2
1c00e0e4:	8d89                	sub	a1,a1,a0
1c00e0e6:	8d99                	sub	a1,a1,a4
1c00e0e8:	c14586b3          	p.extract	a3,a1,0,20
1c00e0ec:	06f00793          	li	a5,111
1c00e0f0:	c1f6a7b3          	p.insert	a5,a3,0,31
1c00e0f4:	d21586b3          	p.extract	a3,a1,9,1
1c00e0f8:	d356a7b3          	p.insert	a5,a3,9,21
1c00e0fc:	c0b586b3          	p.extract	a3,a1,0,11
1c00e100:	c146a7b3          	p.insert	a5,a3,0,20
1c00e104:	cec585b3          	p.extract	a1,a1,7,12
1c00e108:	cec5a7b3          	p.insert	a5,a1,7,12
1c00e10c:	00f56723          	p.sw	a5,a4(a0)
1c00e110:	8082                	ret
1c00e112:	30502773          	csrr	a4,mtvec
1c00e116:	c0073733          	p.bclr	a4,a4,0,0
1c00e11a:	b7e1                	j	1c00e0e2 <pos_irq_set_handler+0x14>

1c00e11c <pos_irq_init>:
1c00e11c:	f1402773          	csrr	a4,mhartid
1c00e120:	1a10a7b7          	lui	a5,0x1a10a
1c00e124:	56fd                	li	a3,-1
1c00e126:	80d7a423          	sw	a3,-2040(a5) # 1a109808 <__l1_end+0xa108e84>
1c00e12a:	ca571733          	p.extractu	a4,a4,5,5
1c00e12e:	1c0087b7          	lui	a5,0x1c008
1c00e132:	46fd                	li	a3,31
1c00e134:	00078793          	mv	a5,a5
1c00e138:	00d70663          	beq	a4,a3,1c00e144 <pos_irq_init+0x28>
1c00e13c:	00200737          	lui	a4,0x200
1c00e140:	c33c                	sw	a5,64(a4)
1c00e142:	8082                	ret
1c00e144:	c007c7b3          	p.bset	a5,a5,0,0
1c00e148:	30579073          	csrw	mtvec,a5
1c00e14c:	8082                	ret

1c00e14e <pos_soc_event_init>:
1c00e14e:	1a1067b7          	lui	a5,0x1a106
1c00e152:	1141                	addi	sp,sp,-16
1c00e154:	c606                	sw	ra,12(sp)
1c00e156:	577d                	li	a4,-1
1c00e158:	00478693          	addi	a3,a5,4 # 1a106004 <__l1_end+0xa105680>
1c00e15c:	c298                	sw	a4,0(a3)
1c00e15e:	00878693          	addi	a3,a5,8
1c00e162:	c298                	sw	a4,0(a3)
1c00e164:	00c78693          	addi	a3,a5,12
1c00e168:	c298                	sw	a4,0(a3)
1c00e16a:	01078693          	addi	a3,a5,16
1c00e16e:	c298                	sw	a4,0(a3)
1c00e170:	01478693          	addi	a3,a5,20
1c00e174:	c298                	sw	a4,0(a3)
1c00e176:	01878693          	addi	a3,a5,24
1c00e17a:	c298                	sw	a4,0(a3)
1c00e17c:	01c78693          	addi	a3,a5,28
1c00e180:	c298                	sw	a4,0(a3)
1c00e182:	02078793          	addi	a5,a5,32
1c00e186:	1c00e5b7          	lui	a1,0x1c00e
1c00e18a:	c398                	sw	a4,0(a5)
1c00e18c:	7a258593          	addi	a1,a1,1954 # 1c00e7a2 <pos_soc_event_handler_asm>
1c00e190:	4569                	li	a0,26
1c00e192:	3f35                	jal	1c00e0ce <pos_irq_set_handler>
1c00e194:	40b2                	lw	ra,12(sp)
1c00e196:	1a10a7b7          	lui	a5,0x1a10a
1c00e19a:	04000737          	lui	a4,0x4000
1c00e19e:	80e7a223          	sw	a4,-2044(a5) # 1a109804 <__l1_end+0xa108e80>
1c00e1a2:	0141                	addi	sp,sp,16
1c00e1a4:	8082                	ret

1c00e1a6 <pos_time_poweroff>:
1c00e1a6:	1a10b7b7          	lui	a5,0x1a10b
1c00e1aa:	0791                	addi	a5,a5,4
1c00e1ac:	0087a783          	lw	a5,8(a5) # 1a10b008 <__l1_end+0xa10a684>
1c00e1b0:	1c002737          	lui	a4,0x1c002
1c00e1b4:	98f72423          	sw	a5,-1656(a4) # 1c001988 <pos_time_timer_count>
1c00e1b8:	4501                	li	a0,0
1c00e1ba:	8082                	ret

1c00e1bc <pos_time_poweron>:
1c00e1bc:	1c0027b7          	lui	a5,0x1c002
1c00e1c0:	9887a703          	lw	a4,-1656(a5) # 1c001988 <pos_time_timer_count>
1c00e1c4:	1a10b7b7          	lui	a5,0x1a10b
1c00e1c8:	0791                	addi	a5,a5,4
1c00e1ca:	00e7a423          	sw	a4,8(a5) # 1a10b008 <__l1_end+0xa10a684>
1c00e1ce:	4501                	li	a0,0
1c00e1d0:	8082                	ret

1c00e1d2 <pos_time_timer_handler>:
1c00e1d2:	1c0028b7          	lui	a7,0x1c002
1c00e1d6:	98888893          	addi	a7,a7,-1656 # 1c001988 <pos_time_timer_count>
1c00e1da:	0048a783          	lw	a5,4(a7)
1c00e1de:	1a10b637          	lui	a2,0x1a10b
1c00e1e2:	0611                	addi	a2,a2,4
1c00e1e4:	00862603          	lw	a2,8(a2) # 1a10b008 <__l1_end+0xa10a684>
1c00e1e8:	cba9                	beqz	a5,1c00e23a <pos_time_timer_handler+0x68>
1c00e1ea:	5fd8                	lw	a4,60(a5)
1c00e1ec:	800005b7          	lui	a1,0x80000
1c00e1f0:	ffe5c593          	xori	a1,a1,-2
1c00e1f4:	40e60733          	sub	a4,a2,a4
1c00e1f8:	0ae5e763          	bltu	a1,a4,1c00e2a6 <pos_time_timer_handler+0xd4>
1c00e1fc:	02002803          	lw	a6,32(zero) # 20 <pos_sched_first>
1c00e200:	02402503          	lw	a0,36(zero) # 24 <pos_sched_last>
1c00e204:	4398                	lw	a4,0(a5)
1c00e206:	0007a023          	sw	zero,0(a5)
1c00e20a:	4301                	li	t1,0
1c00e20c:	02080063          	beqz	a6,1c00e22c <pos_time_timer_handler+0x5a>
1c00e210:	c11c                	sw	a5,0(a0)
1c00e212:	c305                	beqz	a4,1c00e232 <pos_time_timer_handler+0x60>
1c00e214:	5f54                	lw	a3,60(a4)
1c00e216:	853e                	mv	a0,a5
1c00e218:	40d606b3          	sub	a3,a2,a3
1c00e21c:	04d5e763          	bltu	a1,a3,1c00e26a <pos_time_timer_handler+0x98>
1c00e220:	87ba                	mv	a5,a4
1c00e222:	4398                	lw	a4,0(a5)
1c00e224:	0007a023          	sw	zero,0(a5)
1c00e228:	fe0814e3          	bnez	a6,1c00e210 <pos_time_timer_handler+0x3e>
1c00e22c:	883e                	mv	a6,a5
1c00e22e:	4305                	li	t1,1
1c00e230:	f375                	bnez	a4,1c00e214 <pos_time_timer_handler+0x42>
1c00e232:	06031c63          	bnez	t1,1c00e2aa <pos_time_timer_handler+0xd8>
1c00e236:	02f02223          	sw	a5,36(zero) # 24 <pos_sched_last>
1c00e23a:	1a10b7b7          	lui	a5,0x1a10b
1c00e23e:	08100713          	li	a4,129
1c00e242:	0791                	addi	a5,a5,4
1c00e244:	0008a223          	sw	zero,4(a7)
1c00e248:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa10a67c>
1c00e24c:	f14027f3          	csrr	a5,mhartid
1c00e250:	477d                	li	a4,31
1c00e252:	ca5797b3          	p.extractu	a5,a5,5,5
1c00e256:	04e78063          	beq	a5,a4,1c00e296 <pos_time_timer_handler+0xc4>
1c00e25a:	6785                	lui	a5,0x1
1c00e25c:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c00e260:	00204737          	lui	a4,0x204
1c00e264:	02f72423          	sw	a5,40(a4) # 204028 <__l1_heap_size+0x1049ac>
1c00e268:	8082                	ret
1c00e26a:	04031363          	bnez	t1,1c00e2b0 <pos_time_timer_handler+0xde>
1c00e26e:	02f02223          	sw	a5,36(zero) # 24 <pos_sched_last>
1c00e272:	1a10b7b7          	lui	a5,0x1a10b
1c00e276:	0791                	addi	a5,a5,4
1c00e278:	00e8a223          	sw	a4,4(a7)
1c00e27c:	0087a683          	lw	a3,8(a5) # 1a10b008 <__l1_end+0xa10a684>
1c00e280:	5f58                	lw	a4,60(a4)
1c00e282:	40c70633          	sub	a2,a4,a2
1c00e286:	9636                	add	a2,a2,a3
1c00e288:	00c7a823          	sw	a2,16(a5)
1c00e28c:	08500713          	li	a4,133
1c00e290:	00e7a023          	sw	a4,0(a5)
1c00e294:	8082                	ret
1c00e296:	6785                	lui	a5,0x1
1c00e298:	1a10a737          	lui	a4,0x1a10a
1c00e29c:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c00e2a0:	80f72a23          	sw	a5,-2028(a4) # 1a109814 <__l1_end+0xa108e90>
1c00e2a4:	8082                	ret
1c00e2a6:	873e                	mv	a4,a5
1c00e2a8:	b7e9                	j	1c00e272 <pos_time_timer_handler+0xa0>
1c00e2aa:	03002023          	sw	a6,32(zero) # 20 <pos_sched_first>
1c00e2ae:	b761                	j	1c00e236 <pos_time_timer_handler+0x64>
1c00e2b0:	03002023          	sw	a6,32(zero) # 20 <pos_sched_first>
1c00e2b4:	bf6d                	j	1c00e26e <pos_time_timer_handler+0x9c>

1c00e2b6 <pos_time_init>:
1c00e2b6:	1141                	addi	sp,sp,-16
1c00e2b8:	c422                	sw	s0,8(sp)
1c00e2ba:	1a10b7b7          	lui	a5,0x1a10b
1c00e2be:	1c002437          	lui	s0,0x1c002
1c00e2c2:	c606                	sw	ra,12(sp)
1c00e2c4:	98840413          	addi	s0,s0,-1656 # 1c001988 <pos_time_timer_count>
1c00e2c8:	08300713          	li	a4,131
1c00e2cc:	0791                	addi	a5,a5,4
1c00e2ce:	00042223          	sw	zero,4(s0)
1c00e2d2:	00e7a023          	sw	a4,0(a5) # 1a10b000 <__l1_end+0xa10a67c>
1c00e2d6:	1c00e5b7          	lui	a1,0x1c00e
1c00e2da:	77e58593          	addi	a1,a1,1918 # 1c00e77e <pos_time_timer_handler_asm>
1c00e2de:	452d                	li	a0,11
1c00e2e0:	33fd                	jal	1c00e0ce <pos_irq_set_handler>
1c00e2e2:	6785                	lui	a5,0x1
1c00e2e4:	1a10a737          	lui	a4,0x1a10a
1c00e2e8:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c00e2ec:	1c00e637          	lui	a2,0x1c00e
1c00e2f0:	00840513          	addi	a0,s0,8
1c00e2f4:	4681                	li	a3,0
1c00e2f6:	1a660613          	addi	a2,a2,422 # 1c00e1a6 <pos_time_poweroff>
1c00e2fa:	4589                	li	a1,2
1c00e2fc:	80f72223          	sw	a5,-2044(a4) # 1a109804 <__l1_end+0xa108e80>
1c00e300:	36f9                	jal	1c00dece <pos_cbsys_add>
1c00e302:	01840513          	addi	a0,s0,24
1c00e306:	4422                	lw	s0,8(sp)
1c00e308:	40b2                	lw	ra,12(sp)
1c00e30a:	1c00e637          	lui	a2,0x1c00e
1c00e30e:	4681                	li	a3,0
1c00e310:	1bc60613          	addi	a2,a2,444 # 1c00e1bc <pos_time_poweron>
1c00e314:	458d                	li	a1,3
1c00e316:	0141                	addi	sp,sp,16
1c00e318:	be5d                	j	1c00dece <pos_cbsys_add>

1c00e31a <pos_uart_init>:
1c00e31a:	1c0107b7          	lui	a5,0x1c010
1c00e31e:	7807a823          	sw	zero,1936(a5) # 1c010790 <pos_uart>
1c00e322:	8082                	ret

1c00e324 <pi_cluster_conf_init>:
1c00e324:	00052223          	sw	zero,4(a0)
1c00e328:	00052a23          	sw	zero,20(a0)
1c00e32c:	8082                	ret

1c00e32e <pi_cluster_open>:
1c00e32e:	7179                	addi	sp,sp,-48
1c00e330:	c85a                	sw	s6,16(sp)
1c00e332:	00452b03          	lw	s6,4(a0)
1c00e336:	ca56                	sw	s5,20(sp)
1c00e338:	1c0027b7          	lui	a5,0x1c002
1c00e33c:	004b2a83          	lw	s5,4(s6)
1c00e340:	d422                	sw	s0,40(sp)
1c00e342:	c462                	sw	s8,8(sp)
1c00e344:	9e078413          	addi	s0,a5,-1568 # 1c0019e0 <pos_cluster>
1c00e348:	005a9c13          	slli	s8,s5,0x5
1c00e34c:	d606                	sw	ra,44(sp)
1c00e34e:	d226                	sw	s1,36(sp)
1c00e350:	d04a                	sw	s2,32(sp)
1c00e352:	ce4e                	sw	s3,28(sp)
1c00e354:	cc52                	sw	s4,24(sp)
1c00e356:	c65e                	sw	s7,12(sp)
1c00e358:	018404b3          	add	s1,s0,s8
1c00e35c:	c504                	sw	s1,8(a0)
1c00e35e:	1c00e5b7          	lui	a1,0x1c00e
1c00e362:	0ffafb93          	andi	s7,s5,255
1c00e366:	6f658593          	addi	a1,a1,1782 # 1c00e6f6 <pos_task_remote_enqueue>
1c00e36a:	4505                	li	a0,1
1c00e36c:	01748e23          	sb	s7,28(s1)
1c00e370:	3bb9                	jal	1c00e0ce <pos_irq_set_handler>
1c00e372:	040a8793          	addi	a5,s5,64
1c00e376:	1a10a737          	lui	a4,0x1a10a
1c00e37a:	01679993          	slli	s3,a5,0x16
1c00e37e:	00400913          	li	s2,4
1c00e382:	4789                	li	a5,2
1c00e384:	80f72223          	sw	a5,-2044(a4) # 1a109804 <__l1_end+0xa108e80>
1c00e388:	e6c93933          	p.bclr	s2,s2,19,12
1c00e38c:	0004a023          	sw	zero,0(s1)
1c00e390:	994e                	add	s2,s2,s3
1c00e392:	10001a37          	lui	s4,0x10001
1c00e396:	00092023          	sw	zero,0(s2)
1c00e39a:	97ca0a13          	addi	s4,s4,-1668 # 1000097c <pos_cluster_fc_task_lock>
1c00e39e:	4509                	li	a0,2
1c00e3a0:	00092623          	sw	zero,12(s2)
1c00e3a4:	00092223          	sw	zero,4(s2)
1c00e3a8:	000a2023          	sw	zero,0(s4)
1c00e3ac:	cd3fd0ef          	jal	ra,1c00c07e <pos_fll_init>
1c00e3b0:	01400793          	li	a5,20
1c00e3b4:	478c                	lw	a1,8(a5)
1c00e3b6:	ede1                	bnez	a1,1c00e48e <pi_cluster_open+0x160>
1c00e3b8:	c788                	sw	a0,8(a5)
1c00e3ba:	002004b7          	lui	s1,0x200
1c00e3be:	94ce                	add	s1,s1,s3
1c00e3c0:	4785                	li	a5,1
1c00e3c2:	02f4a023          	sw	a5,32(s1) # 200020 <__l1_heap_size+0x1009a4>
1c00e3c6:	00400693          	li	a3,4
1c00e3ca:	6785                	lui	a5,0x1
1c00e3cc:	e6c6b6b3          	p.bclr	a3,a3,19,12
1c00e3d0:	98078793          	addi	a5,a5,-1664 # 980 <_l1_preload_size>
1c00e3d4:	96ce                	add	a3,a3,s3
1c00e3d6:	02f05663          	blez	a5,1c00e402 <pi_cluster_open+0xd4>
1c00e3da:	17fd                	addi	a5,a5,-1
1c00e3dc:	1c010737          	lui	a4,0x1c010
1c00e3e0:	6605                	lui	a2,0x1
1c00e3e2:	7c870713          	addi	a4,a4,1992 # 1c0107c8 <_l1_preload_start_inL2>
1c00e3e6:	8389                	srli	a5,a5,0x2
1c00e3e8:	97c60613          	addi	a2,a2,-1668 # 97c <pos_soc_event_callback+0x534>
1c00e3ec:	55f5                	li	a1,-3
1c00e3ee:	8e99                	sub	a3,a3,a4
1c00e3f0:	0785                	addi	a5,a5,1
1c00e3f2:	0ab64263          	blt	a2,a1,1c00e496 <pi_cluster_open+0x168>
1c00e3f6:	0057c0fb          	lp.setup	x1,a5,1c00e400 <pi_cluster_open+0xd2>
1c00e3fa:	4310                	lw	a2,0(a4)
1c00e3fc:	00c766a3          	p.sw	a2,a3(a4)
1c00e400:	0711                	addi	a4,a4,4
1c00e402:	018407b3          	add	a5,s0,s8
1c00e406:	01778e23          	sb	s7,28(a5)
1c00e40a:	014b4683          	lbu	a3,20(s6)
1c00e40e:	00201437          	lui	s0,0x201
1c00e412:	e0440713          	addi	a4,s0,-508 # 200e04 <__l1_heap_size+0x101788>
1c00e416:	974e                	add	a4,a4,s3
1c00e418:	cf98                	sw	a4,24(a5)
1c00e41a:	cb98                	sw	a4,16(a5)
1c00e41c:	0127a223          	sw	s2,4(a5)
1c00e420:	00d78ea3          	sb	a3,29(a5)
1c00e424:	8556                	mv	a0,s5
1c00e426:	0007a423          	sw	zero,8(a5)
1c00e42a:	0007aa23          	sw	zero,20(a5)
1c00e42e:	00078f23          	sb	zero,30(a5)
1c00e432:	00da0223          	sb	a3,4(s4)
1c00e436:	36d5                	jal	1c00e01a <pos_alloc_init_l1>
1c00e438:	40040793          	addi	a5,s0,1024
1c00e43c:	97ce                	add	a5,a5,s3
1c00e43e:	577d                	li	a4,-1
1c00e440:	00e7a023          	sw	a4,0(a5)
1c00e444:	1c0087b7          	lui	a5,0x1c008
1c00e448:	08078793          	addi	a5,a5,128 # 1c008080 <_start>
1c00e44c:	04f4a023          	sw	a5,64(s1)
1c00e450:	04f4a223          	sw	a5,68(s1)
1c00e454:	04f4a423          	sw	a5,72(s1)
1c00e458:	04f4a623          	sw	a5,76(s1)
1c00e45c:	04f4a823          	sw	a5,80(s1)
1c00e460:	04f4aa23          	sw	a5,84(s1)
1c00e464:	04f4ac23          	sw	a5,88(s1)
1c00e468:	04f4ae23          	sw	a5,92(s1)
1c00e46c:	0ff00793          	li	a5,255
1c00e470:	00f4a423          	sw	a5,8(s1)
1c00e474:	50b2                	lw	ra,44(sp)
1c00e476:	5422                	lw	s0,40(sp)
1c00e478:	5492                	lw	s1,36(sp)
1c00e47a:	5902                	lw	s2,32(sp)
1c00e47c:	49f2                	lw	s3,28(sp)
1c00e47e:	4a62                	lw	s4,24(sp)
1c00e480:	4ad2                	lw	s5,20(sp)
1c00e482:	4b42                	lw	s6,16(sp)
1c00e484:	4bb2                	lw	s7,12(sp)
1c00e486:	4c22                	lw	s8,8(sp)
1c00e488:	4501                	li	a0,0
1c00e48a:	6145                	addi	sp,sp,48
1c00e48c:	8082                	ret
1c00e48e:	4509                	li	a0,2
1c00e490:	b95fd0ef          	jal	ra,1c00c024 <pos_fll_set_freq>
1c00e494:	b71d                	j	1c00e3ba <pi_cluster_open+0x8c>
1c00e496:	4785                	li	a5,1
1c00e498:	bfb9                	j	1c00e3f6 <pi_cluster_open+0xc8>

1c00e49a <pi_cluster_close>:
1c00e49a:	4501                	li	a0,0
1c00e49c:	8082                	ret

1c00e49e <pos_cluster_task_set_stack>:
1c00e49e:	1101                	addi	sp,sp,-32
1c00e4a0:	c84a                	sw	s2,16(sp)
1c00e4a2:	00852903          	lw	s2,8(a0)
1c00e4a6:	ca26                	sw	s1,20(sp)
1c00e4a8:	c64e                	sw	s3,12(sp)
1c00e4aa:	4785                	li	a5,1
1c00e4ac:	ce06                	sw	ra,28(sp)
1c00e4ae:	cc22                	sw	s0,24(sp)
1c00e4b0:	00f90f23          	sb	a5,30(s2)
1c00e4b4:	45dc                	lw	a5,12(a1)
1c00e4b6:	89aa                	mv	s3,a0
1c00e4b8:	84ae                	mv	s1,a1
1c00e4ba:	ebb1                	bnez	a5,1c00e50e <pos_cluster_task_set_stack+0x70>
1c00e4bc:	6785                	lui	a5,0x1
1c00e4be:	80078793          	addi	a5,a5,-2048 # 800 <pos_soc_event_callback+0x3b8>
1c00e4c2:	40000713          	li	a4,1024
1c00e4c6:	c5dc                	sw	a5,12(a1)
1c00e4c8:	c998                	sw	a4,16(a1)
1c00e4ca:	40000693          	li	a3,1024
1c00e4ce:	48d8                	lw	a4,20(s1)
1c00e4d0:	00892503          	lw	a0,8(s2)
1c00e4d4:	843e                	mv	s0,a5
1c00e4d6:	177d                	addi	a4,a4,-1
1c00e4d8:	42d70433          	p.mac	s0,a4,a3
1c00e4dc:	c901                	beqz	a0,1c00e4ec <pos_cluster_task_set_stack+0x4e>
1c00e4de:	00c92603          	lw	a2,12(s2)
1c00e4e2:	00860d63          	beq	a2,s0,1c00e4fc <pos_cluster_task_set_stack+0x5e>
1c00e4e6:	85aa                	mv	a1,a0
1c00e4e8:	854e                	mv	a0,s3
1c00e4ea:	36f1                	jal	1c00e0b6 <pi_cl_l1_free>
1c00e4ec:	00892623          	sw	s0,12(s2)
1c00e4f0:	85a2                	mv	a1,s0
1c00e4f2:	854e                	mv	a0,s3
1c00e4f4:	366d                	jal	1c00e09e <pi_cl_l1_malloc>
1c00e4f6:	00a92423          	sw	a0,8(s2)
1c00e4fa:	cd19                	beqz	a0,1c00e518 <pos_cluster_task_set_stack+0x7a>
1c00e4fc:	c488                	sw	a0,8(s1)
1c00e4fe:	4501                	li	a0,0
1c00e500:	40f2                	lw	ra,28(sp)
1c00e502:	4462                	lw	s0,24(sp)
1c00e504:	44d2                	lw	s1,20(sp)
1c00e506:	4942                	lw	s2,16(sp)
1c00e508:	49b2                	lw	s3,12(sp)
1c00e50a:	6105                	addi	sp,sp,32
1c00e50c:	8082                	ret
1c00e50e:	4994                	lw	a3,16(a1)
1c00e510:	fedd                	bnez	a3,1c00e4ce <pos_cluster_task_set_stack+0x30>
1c00e512:	c99c                	sw	a5,16(a1)
1c00e514:	86be                	mv	a3,a5
1c00e516:	bf65                	j	1c00e4ce <pos_cluster_task_set_stack+0x30>
1c00e518:	557d                	li	a0,-1
1c00e51a:	b7dd                	j	1c00e500 <pos_cluster_task_set_stack+0x62>

1c00e51c <pi_cluster_send_task_to_cl>:
1c00e51c:	715d                	addi	sp,sp,-80
1c00e51e:	c2a6                	sw	s1,68(sp)
1c00e520:	4504                	lw	s1,8(a0)
1c00e522:	c4a2                	sw	s0,72(sp)
1c00e524:	c0ca                	sw	s2,64(sp)
1c00e526:	c686                	sw	ra,76(sp)
1c00e528:	1c00e7b7          	lui	a5,0x1c00e
1c00e52c:	eec78793          	addi	a5,a5,-276 # 1c00deec <pos_task_handle_blocking>
1c00e530:	01d4c703          	lbu	a4,29(s1)
1c00e534:	c23e                	sw	a5,4(sp)
1c00e536:	00010793          	mv	a5,sp
1c00e53a:	00010a23          	sb	zero,20(sp)
1c00e53e:	c43e                	sw	a5,8(sp)
1c00e540:	842e                	mv	s0,a1
1c00e542:	0044a903          	lw	s2,4(s1)
1c00e546:	459c                	lw	a5,8(a1)
1c00e548:	04173c63          	p.bneimm	a4,1,1c00e5a0 <pi_cluster_send_task_to_cl+0x84>
1c00e54c:	c3b5                	beqz	a5,1c00e5b0 <pi_cluster_send_task_to_cl+0x94>
1c00e54e:	00010793          	mv	a5,sp
1c00e552:	02042023          	sw	zero,32(s0)
1c00e556:	cc1c                	sw	a5,24(s0)
1c00e558:	30047773          	csrrci	a4,mstatus,8
1c00e55c:	409c                	lw	a5,0(s1)
1c00e55e:	c391                	beqz	a5,1c00e562 <pi_cluster_send_task_to_cl+0x46>
1c00e560:	d380                	sw	s0,32(a5)
1c00e562:	c080                	sw	s0,0(s1)
1c00e564:	00092783          	lw	a5,0(s2)
1c00e568:	cb8d                	beqz	a5,1c00e59a <pi_cluster_send_task_to_cl+0x7e>
1c00e56a:	4c9c                	lw	a5,24(s1)
1c00e56c:	0007a023          	sw	zero,0(a5)
1c00e570:	30071073          	csrw	mstatus,a4
1c00e574:	30047473          	csrrci	s0,mstatus,8
1c00e578:	01410783          	lb	a5,20(sp)
1c00e57c:	e791                	bnez	a5,1c00e588 <pi_cluster_send_task_to_cl+0x6c>
1c00e57e:	977ff0ef          	jal	ra,1c00def4 <pos_task_handle>
1c00e582:	01410783          	lb	a5,20(sp)
1c00e586:	dfe5                	beqz	a5,1c00e57e <pi_cluster_send_task_to_cl+0x62>
1c00e588:	30041073          	csrw	mstatus,s0
1c00e58c:	4501                	li	a0,0
1c00e58e:	40b6                	lw	ra,76(sp)
1c00e590:	4426                	lw	s0,72(sp)
1c00e592:	4496                	lw	s1,68(sp)
1c00e594:	4906                	lw	s2,64(sp)
1c00e596:	6161                	addi	sp,sp,80
1c00e598:	8082                	ret
1c00e59a:	00892023          	sw	s0,0(s2)
1c00e59e:	b7f1                	j	1c00e56a <pi_cluster_send_task_to_cl+0x4e>
1c00e5a0:	cf99                	beqz	a5,1c00e5be <pi_cluster_send_task_to_cl+0xa2>
1c00e5a2:	4858                	lw	a4,20(s0)
1c00e5a4:	4785                	li	a5,1
1c00e5a6:	00e797b3          	sll	a5,a5,a4
1c00e5aa:	17fd                	addi	a5,a5,-1
1c00e5ac:	d05c                	sw	a5,36(s0)
1c00e5ae:	b745                	j	1c00e54e <pi_cluster_send_task_to_cl+0x32>
1c00e5b0:	01e4c783          	lbu	a5,30(s1)
1c00e5b4:	ffc9                	bnez	a5,1c00e54e <pi_cluster_send_task_to_cl+0x32>
1c00e5b6:	35e5                	jal	1c00e49e <pos_cluster_task_set_stack>
1c00e5b8:	d959                	beqz	a0,1c00e54e <pi_cluster_send_task_to_cl+0x32>
1c00e5ba:	557d                	li	a0,-1
1c00e5bc:	bfc9                	j	1c00e58e <pi_cluster_send_task_to_cl+0x72>
1c00e5be:	35c5                	jal	1c00e49e <pos_cluster_task_set_stack>
1c00e5c0:	d16d                	beqz	a0,1c00e5a2 <pi_cluster_send_task_to_cl+0x86>
1c00e5c2:	bfe5                	j	1c00e5ba <pi_cluster_send_task_to_cl+0x9e>

1c00e5c4 <pos_init_entry>:
1c00e5c4:	7a101073          	csrw	pcmr,zero
1c00e5c8:	f1402573          	csrr	a0,mhartid
1c00e5cc:	01f57593          	andi	a1,a0,31
1c00e5d0:	8115                	srli	a0,a0,0x5
1c00e5d2:	467d                	li	a2,31
1c00e5d4:	00c50463          	beq	a0,a2,1c00e5dc <pos_init_entry+0x18>
1c00e5d8:	2290106f          	j	1c010000 <__cluster_text_start>
1c00e5dc:	ffff3297          	auipc	t0,0xffff3
1c00e5e0:	3d428293          	addi	t0,t0,980 # 1c0019b0 <_edata>
1c00e5e4:	ffff3317          	auipc	t1,0xffff3
1c00e5e8:	41c30313          	addi	t1,t1,1052 # 1c001a00 <__l2_priv0_end>
1c00e5ec:	0002a023          	sw	zero,0(t0)
1c00e5f0:	0291                	addi	t0,t0,4
1c00e5f2:	fe62ede3          	bltu	t0,t1,1c00e5ec <pos_init_entry+0x28>
1c00e5f6:	ffff3117          	auipc	sp,0xffff3
1c00e5fa:	37a10113          	addi	sp,sp,890 # 1c001970 <stack>
1c00e5fe:	00000297          	auipc	t0,0x0
1c00e602:	87028293          	addi	t0,t0,-1936 # 1c00de6e <pos_init_start>
1c00e606:	000280e7          	jalr	t0
1c00e60a:	00000513          	li	a0,0
1c00e60e:	00000593          	li	a1,0
1c00e612:	ffffd397          	auipc	t2,0xffffd
1c00e616:	d3238393          	addi	t2,t2,-718 # 1c00b344 <main>
1c00e61a:	000380e7          	jalr	t2
1c00e61e:	842a                	mv	s0,a0
1c00e620:	8522                	mv	a0,s0
1c00e622:	ffffe297          	auipc	t0,0xffffe
1c00e626:	c3a28293          	addi	t0,t0,-966 # 1c00c25c <exit>
1c00e62a:	000280e7          	jalr	t0

1c00e62e <pos_irq_call_external_c_function>:
1c00e62e:	7119                	addi	sp,sp,-128
1c00e630:	c006                	sw	ra,0(sp)
1c00e632:	c20e                	sw	gp,4(sp)
1c00e634:	c412                	sw	tp,8(sp)
1c00e636:	c616                	sw	t0,12(sp)
1c00e638:	c81a                	sw	t1,16(sp)
1c00e63a:	ca1e                	sw	t2,20(sp)
1c00e63c:	d236                	sw	a3,36(sp)
1c00e63e:	d43a                	sw	a4,40(sp)
1c00e640:	d63e                	sw	a5,44(sp)
1c00e642:	d842                	sw	a6,48(sp)
1c00e644:	da46                	sw	a7,52(sp)
1c00e646:	dc72                	sw	t3,56(sp)
1c00e648:	de76                	sw	t4,60(sp)
1c00e64a:	c0fa                	sw	t5,64(sp)
1c00e64c:	c6fe                	sw	t6,76(sp)
1c00e64e:	000600e7          	jalr	a2
1c00e652:	4082                	lw	ra,0(sp)
1c00e654:	4192                	lw	gp,4(sp)
1c00e656:	4222                	lw	tp,8(sp)
1c00e658:	42b2                	lw	t0,12(sp)
1c00e65a:	4342                	lw	t1,16(sp)
1c00e65c:	43d2                	lw	t2,20(sp)
1c00e65e:	5692                	lw	a3,36(sp)
1c00e660:	5722                	lw	a4,40(sp)
1c00e662:	57b2                	lw	a5,44(sp)
1c00e664:	5842                	lw	a6,48(sp)
1c00e666:	58d2                	lw	a7,52(sp)
1c00e668:	5e62                	lw	t3,56(sp)
1c00e66a:	5ef2                	lw	t4,60(sp)
1c00e66c:	4f06                	lw	t5,64(sp)
1c00e66e:	4fb6                	lw	t6,76(sp)
1c00e670:	6109                	addi	sp,sp,128
1c00e672:	8482                	jr	s1

1c00e674 <pos_irq_call_external_c_function_full>:
1c00e674:	7119                	addi	sp,sp,-128
1c00e676:	c006                	sw	ra,0(sp)
1c00e678:	c20e                	sw	gp,4(sp)
1c00e67a:	c412                	sw	tp,8(sp)
1c00e67c:	c616                	sw	t0,12(sp)
1c00e67e:	c81a                	sw	t1,16(sp)
1c00e680:	ca1e                	sw	t2,20(sp)
1c00e682:	cc2a                	sw	a0,24(sp)
1c00e684:	ce2e                	sw	a1,28(sp)
1c00e686:	d236                	sw	a3,36(sp)
1c00e688:	d43a                	sw	a4,40(sp)
1c00e68a:	d63e                	sw	a5,44(sp)
1c00e68c:	d842                	sw	a6,48(sp)
1c00e68e:	da46                	sw	a7,52(sp)
1c00e690:	dc72                	sw	t3,56(sp)
1c00e692:	de76                	sw	t4,60(sp)
1c00e694:	c0fa                	sw	t5,64(sp)
1c00e696:	c6fe                	sw	t6,76(sp)
1c00e698:	000600e7          	jalr	a2
1c00e69c:	4082                	lw	ra,0(sp)
1c00e69e:	4192                	lw	gp,4(sp)
1c00e6a0:	4222                	lw	tp,8(sp)
1c00e6a2:	42b2                	lw	t0,12(sp)
1c00e6a4:	4342                	lw	t1,16(sp)
1c00e6a6:	43d2                	lw	t2,20(sp)
1c00e6a8:	4562                	lw	a0,24(sp)
1c00e6aa:	45f2                	lw	a1,28(sp)
1c00e6ac:	5692                	lw	a3,36(sp)
1c00e6ae:	5722                	lw	a4,40(sp)
1c00e6b0:	57b2                	lw	a5,44(sp)
1c00e6b2:	5842                	lw	a6,48(sp)
1c00e6b4:	58d2                	lw	a7,52(sp)
1c00e6b6:	5e62                	lw	t3,56(sp)
1c00e6b8:	5ef2                	lw	t4,60(sp)
1c00e6ba:	4f06                	lw	t5,64(sp)
1c00e6bc:	4fb6                	lw	t6,76(sp)
1c00e6be:	6109                	addi	sp,sp,128
1c00e6c0:	8482                	jr	s1

1c00e6c2 <pos_task_push_asm>:
1c00e6c2:	e3ff2297          	auipc	t0,0xe3ff2
1c00e6c6:	95e28293          	addi	t0,t0,-1698 # 20 <pos_sched_first>
1c00e6ca:	0002a603          	lw	a2,0(t0)
1c00e6ce:	0005a023          	sw	zero,0(a1)
1c00e6d2:	ca09                	beqz	a2,1c00e6e4 <__rt_no_first>
1c00e6d4:	e3ff2297          	auipc	t0,0xe3ff2
1c00e6d8:	95028293          	addi	t0,t0,-1712 # 24 <pos_sched_last>
1c00e6dc:	0002a603          	lw	a2,0(t0)
1c00e6e0:	c20c                	sw	a1,0(a2)
1c00e6e2:	a029                	j	1c00e6ec <__rt_common>

1c00e6e4 <__rt_no_first>:
1c00e6e4:	e3ff2297          	auipc	t0,0xe3ff2
1c00e6e8:	92b2ae23          	sw	a1,-1732(t0) # 20 <pos_sched_first>

1c00e6ec <__rt_common>:
1c00e6ec:	e3ff2297          	auipc	t0,0xe3ff2
1c00e6f0:	92b2ac23          	sw	a1,-1736(t0) # 24 <pos_sched_last>

1c00e6f4 <enqueue_end>:
1c00e6f4:	8482                	jr	s1

1c00e6f6 <pos_task_remote_enqueue>:
1c00e6f6:	fe812e23          	sw	s0,-4(sp)
1c00e6fa:	fe912c23          	sw	s1,-8(sp)
1c00e6fe:	fea12a23          	sw	a0,-12(sp)
1c00e702:	feb12823          	sw	a1,-16(sp)
1c00e706:	fec12623          	sw	a2,-20(sp)
1c00e70a:	4405                	li	s0,1
1c00e70c:	ffff3497          	auipc	s1,0xffff3
1c00e710:	2d448493          	addi	s1,s1,724 # 1c0019e0 <pos_cluster>

1c00e714 <pos_task_remote_enqueue_event_loop_cluster>:
1c00e714:	48cc                	lw	a1,20(s1)
1c00e716:	40d0                	lw	a2,4(s1)
1c00e718:	02058a63          	beqz	a1,1c00e74c <pos_task_remote_enqueue_event_loop_cluster_continue>
1c00e71c:	4208                	lw	a0,0(a2)
1c00e71e:	e119                	bnez	a0,1c00e724 <pos_task_remote_enqueue_no_queue_update>
1c00e720:	0004a023          	sw	zero,0(s1)

1c00e724 <pos_task_remote_enqueue_no_queue_update>:
1c00e724:	4890                	lw	a2,16(s1)
1c00e726:	0004aa23          	sw	zero,20(s1)
1c00e72a:	00062023          	sw	zero,0(a2)
1c00e72e:	00000497          	auipc	s1,0x0
1c00e732:	01e48493          	addi	s1,s1,30 # 1c00e74c <pos_task_remote_enqueue_event_loop_cluster_continue>
1c00e736:	0035f513          	andi	a0,a1,3
1c00e73a:	00051363          	bnez	a0,1c00e740 <pos_task_callback>
1c00e73e:	b751                	j	1c00e6c2 <pos_task_push_asm>

1c00e740 <pos_task_callback>:
1c00e740:	5571                	li	a0,-4
1c00e742:	8de9                	and	a1,a1,a0
1c00e744:	41d0                	lw	a2,4(a1)
1c00e746:	4588                	lw	a0,8(a1)
1c00e748:	ee7ff06f          	j	1c00e62e <pos_irq_call_external_c_function>

1c00e74c <pos_task_remote_enqueue_event_loop_cluster_continue>:
1c00e74c:	147d                	addi	s0,s0,-1
1c00e74e:	00804e63          	bgtz	s0,1c00e76a <pos_task_remote_enqueue_event_loop_next_cluster>
1c00e752:	ffc12403          	lw	s0,-4(sp)
1c00e756:	ff812483          	lw	s1,-8(sp)
1c00e75a:	ff412503          	lw	a0,-12(sp)
1c00e75e:	ff012583          	lw	a1,-16(sp)
1c00e762:	fec12603          	lw	a2,-20(sp)
1c00e766:	30200073          	mret

1c00e76a <pos_task_remote_enqueue_event_loop_next_cluster>:
1c00e76a:	ffff3497          	auipc	s1,0xffff3
1c00e76e:	27648493          	addi	s1,s1,630 # 1c0019e0 <pos_cluster>
1c00e772:	02c00593          	li	a1,44
1c00e776:	02b405b3          	mul	a1,s0,a1
1c00e77a:	94ae                	add	s1,s1,a1
1c00e77c:	bf61                	j	1c00e714 <pos_task_remote_enqueue_event_loop_cluster>

1c00e77e <pos_time_timer_handler_asm>:
1c00e77e:	1161                	addi	sp,sp,-8
1c00e780:	c032                	sw	a2,0(sp)
1c00e782:	c226                	sw	s1,4(sp)
1c00e784:	00000617          	auipc	a2,0x0
1c00e788:	a4e60613          	addi	a2,a2,-1458 # 1c00e1d2 <pos_time_timer_handler>
1c00e78c:	00000497          	auipc	s1,0x0
1c00e790:	00c48493          	addi	s1,s1,12 # 1c00e798 <pos_time_timer_handler_asm_ret>
1c00e794:	ee1ff06f          	j	1c00e674 <pos_irq_call_external_c_function_full>

1c00e798 <pos_time_timer_handler_asm_ret>:
1c00e798:	4492                	lw	s1,4(sp)
1c00e79a:	4602                	lw	a2,0(sp)
1c00e79c:	0121                	addi	sp,sp,8
1c00e79e:	30200073          	mret

1c00e7a2 <pos_soc_event_handler_asm>:
1c00e7a2:	7119                	addi	sp,sp,-128
1c00e7a4:	c022                	sw	s0,0(sp)
1c00e7a6:	c226                	sw	s1,4(sp)
1c00e7a8:	c42a                	sw	a0,8(sp)
1c00e7aa:	c62e                	sw	a1,12(sp)
1c00e7ac:	c832                	sw	a2,16(sp)
1c00e7ae:	1a10a437          	lui	s0,0x1a10a
1c00e7b2:	80040413          	addi	s0,s0,-2048 # 1a109800 <__l1_end+0xa108e7c>
1c00e7b6:	5048                	lw	a0,36(s0)
1c00e7b8:	00251593          	slli	a1,a0,0x2
1c00e7bc:	e3ff2297          	auipc	t0,0xe3ff2
1c00e7c0:	c8c28293          	addi	t0,t0,-884 # 448 <pos_soc_event_callback>
1c00e7c4:	2055f603          	p.lw	a2,t0(a1)
1c00e7c8:	e3ff2297          	auipc	t0,0xe3ff2
1c00e7cc:	88028293          	addi	t0,t0,-1920 # 48 <pos_soc_event_callback_arg>
1c00e7d0:	2055f583          	p.lw	a1,t0(a1)
1c00e7d4:	00000497          	auipc	s1,0x0
1c00e7d8:	03048493          	addi	s1,s1,48 # 1c00e804 <pos_soc_event_handler_end_asm>
1c00e7dc:	e53ff06f          	j	1c00e62e <pos_irq_call_external_c_function>

1c00e7e0 <pos_soc_event_no_udma_asm>:
1c00e7e0:	00555593          	srli	a1,a0,0x5
1c00e7e4:	058a                	slli	a1,a1,0x2
1c00e7e6:	e3ff2297          	auipc	t0,0xe3ff2
1c00e7ea:	84228293          	addi	t0,t0,-1982 # 28 <pos_soc_event_status>
1c00e7ee:	2055f603          	p.lw	a2,t0(a1)
1c00e7f2:	897d                	andi	a0,a0,31
1c00e7f4:	80a64633          	p.bsetr	a2,a2,a0
1c00e7f8:	e3ff2297          	auipc	t0,0xe3ff2
1c00e7fc:	83028293          	addi	t0,t0,-2000 # 28 <pos_soc_event_status>
1c00e800:	00c5e2a3          	p.sw	a2,t0(a1)

1c00e804 <pos_soc_event_handler_end_asm>:
1c00e804:	4402                	lw	s0,0(sp)
1c00e806:	4492                	lw	s1,4(sp)
1c00e808:	4522                	lw	a0,8(sp)
1c00e80a:	45b2                	lw	a1,12(sp)
1c00e80c:	4642                	lw	a2,16(sp)
1c00e80e:	6109                	addi	sp,sp,128
1c00e810:	30200073          	mret

Disassembly of section .l2_data:

1c010000 <__cluster_text_start>:
1c010000:	f1402573          	csrr	a0,mhartid
1c010004:	01f57593          	andi	a1,a0,31
1c010008:	8115                	srli	a0,a0,0x5
1c01000a:	000702b7          	lui	t0,0x70
1c01000e:	00204337          	lui	t1,0x204
1c010012:	00532023          	sw	t0,0(t1) # 204000 <__l1_heap_size+0x104984>
1c010016:	4381                	li	t2,0
1c010018:	0e759463          	bne	a1,t2,1c010100 <pos_slave_start>
1c01001c:	20000293          	li	t0,512
1c010020:	00204337          	lui	t1,0x204
1c010024:	00532a23          	sw	t0,20(t1) # 204014 <__l1_heap_size+0x104998>
1c010028:	e3ff0417          	auipc	s0,0xe3ff0
1c01002c:	fdc40413          	addi	s0,s0,-36 # 4 <pos_fll_freq>
1c010030:	002049b7          	lui	s3,0x204
1c010034:	4a09                	li	s4,2
1c010036:	00000a97          	auipc	s5,0x0
1c01003a:	038a8a93          	addi	s5,s5,56 # 1c01006e <pos_master_event>
1c01003e:	ffff2b97          	auipc	s7,0xffff2
1c010042:	9a2b8b93          	addi	s7,s7,-1630 # 1c0019e0 <pos_cluster>
1c010046:	02c00393          	li	t2,44
1c01004a:	02a383b3          	mul	t2,t2,a0
1c01004e:	9b9e                	add	s7,s7,t2
1c010050:	0bd1                	addi	s7,s7,20
1c010052:	1a10acb7          	lui	s9,0x1a10a
1c010056:	810c8c93          	addi	s9,s9,-2032 # 1a109810 <__l1_end+0xa108e8c>
1c01005a:	4c09                	li	s8,2
1c01005c:	00000d17          	auipc	s10,0x0
1c010060:	0e2d0d13          	addi	s10,s10,226 # 1c01013e <pos_set_slave_stack>
1c010064:	001d6d13          	ori	s10,s10,1
1c010068:	30045073          	csrwi	mstatus,8
1c01006c:	a829                	j	1c010086 <pos_master_loop>

1c01006e <pos_master_event>:
1c01006e:	018b2b03          	lw	s6,24(s6)
1c010072:	000b0a63          	beqz	s6,1c010086 <pos_master_loop>

1c010076 <pos_push_event_to_fc_retry>:
1c010076:	000ba283          	lw	t0,0(s7)
1c01007a:	06029c63          	bnez	t0,1c0100f2 <pos_push_event_to_fc_wait>
1c01007e:	016ba023          	sw	s6,0(s7)
1c010082:	018ca023          	sw	s8,0(s9)

1c010086 <pos_master_loop>:
1c010086:	00042b03          	lw	s6,0(s0)
1c01008a:	040b0d63          	beqz	s6,1c0100e4 <pos_master_sleep>

1c01008e <pos_master_loop_update_next>:
1c01008e:	020b2e83          	lw	t4,32(s6)
1c010092:	01d42023          	sw	t4,0(s0)
1c010096:	020b2f03          	lw	t5,32(s6)
1c01009a:	ffee9ae3          	bne	t4,t5,1c01008e <pos_master_loop_update_next>

1c01009e <pos_master_loop_exec_task>:
1c01009e:	004b2503          	lw	a0,4(s6)
1c0100a2:	000b2283          	lw	t0,0(s6)
1c0100a6:	008b2103          	lw	sp,8(s6)
1c0100aa:	00cb2303          	lw	t1,12(s6)
1c0100ae:	010b2383          	lw	t2,16(s6)
1c0100b2:	024b2f03          	lw	t5,36(s6)
1c0100b6:	014b2f83          	lw	t6,20(s6)
1c0100ba:	80d6                	mv	ra,s5
1c0100bc:	911a                	add	sp,sp,t1
1c0100be:	e3ff0e17          	auipc	t3,0xe3ff0
1c0100c2:	f5fe2d23          	sw	t6,-166(t3) # 18 <pos_cluster_nb_active_pe>

1c0100c6 <pos_no_stack_check>:
1c0100c6:	09e9a223          	sw	t5,132(s3) # 204084 <__l1_heap_size+0x104a08>
1c0100ca:	21e9a023          	sw	t5,512(s3)
1c0100ce:	21e9a623          	sw	t5,524(s3)

1c0100d2 <pos_master_no_slave_barrier>:
1c0100d2:	000f2863          	p.beqimm	t5,0,1c0100e2 <pos_master_loop_no_slave>
1c0100d6:	09a9a023          	sw	s10,128(s3)
1c0100da:	0879a023          	sw	t2,128(s3)
1c0100de:	0829a023          	sw	sp,128(s3)

1c0100e2 <pos_master_loop_no_slave>:
1c0100e2:	8282                	jr	t0

1c0100e4 <pos_master_sleep>:
1c0100e4:	0149a423          	sw	s4,8(s3)
1c0100e8:	03c9e003          	p.elw	zero,60(s3)
1c0100ec:	0149a223          	sw	s4,4(s3)
1c0100f0:	bf59                	j	1c010086 <pos_master_loop>

1c0100f2 <pos_push_event_to_fc_wait>:
1c0100f2:	0149a423          	sw	s4,8(s3)
1c0100f6:	03c9e003          	p.elw	zero,60(s3)
1c0100fa:	0149a223          	sw	s4,4(s3)
1c0100fe:	bfa5                	j	1c010076 <pos_push_event_to_fc_retry>

1c010100 <pos_slave_start>:
1c010100:	00204937          	lui	s2,0x204
1c010104:	f14029f3          	csrr	s3,mhartid
1c010108:	01f9f993          	andi	s3,s3,31
1c01010c:	00000a17          	auipc	s4,0x0
1c010110:	012a0a13          	addi	s4,s4,18 # 1c01011e <pos_fork_return>
1c010114:	00000a97          	auipc	s5,0x0
1c010118:	00ea8a93          	addi	s5,s5,14 # 1c010122 <pos_wait_for_dispatch>
1c01011c:	a019                	j	1c010122 <pos_wait_for_dispatch>

1c01011e <pos_fork_return>:
1c01011e:	21c96283          	p.elw	t0,540(s2) # 20421c <__l1_heap_size+0x104ba0>

1c010122 <pos_wait_for_dispatch>:
1c010122:	08096283          	p.elw	t0,128(s2)
1c010126:	08096503          	p.elw	a0,128(s2)
1c01012a:	0012f313          	andi	t1,t0,1
1c01012e:	00031563          	bnez	t1,1c010138 <pos_other_entry>

1c010132 <pos_fork_entry>:
1c010132:	000a00b3          	add	ra,s4,zero
1c010136:	8282                	jr	t0

1c010138 <pos_other_entry>:
1c010138:	000a80b3          	add	ra,s5,zero
1c01013c:	8282                	jr	t0

1c01013e <pos_set_slave_stack>:
1c01013e:	08096283          	p.elw	t0,128(s2)
1c010142:	00098f13          	mv	t5,s3
1c010146:	02af0eb3          	mul	t4,t5,a0
1c01014a:	005e8133          	add	sp,t4,t0
1c01014e:	8082                	ret

1c010150 <gm_cos>:
1c010150:	67e8 bf11 6c21 3f19 1f72 bf32 b1e5 3f63     .g..!l.?r.2...c?
1c010160:	8ffc 3f57 b29a 3f6a 8652 bee0 1586 3f7a     ..W?..j?R.....z?
1c010170:	eb02 3bc1 7ece 3ed9 a544 bf0b 3018 bf7d     ...;.~.>D....0}.
1c010180:	6f41 bebb b8b2 3d1d 3d94 3f34 d238 be35     Ao.....=.=4?8.5.
1c010190:	4490 3dbf dfab bf7f 9fd6 bebe aca0 3f6e     .D.=..........n?
1c0101a0:	3da0 3f40 7142 bf5a b4c2 bf7f 2bec 3ef0     .=@?BqZ......+.>
1c0101b0:	dfdd 3f61 4357 bf50 72e1 3f7e 178e bd96     ..a?WCP..r~?....
1c0101c0:	b667 3e78 9bc7 3f69 b98c bf7e fb3c bf7f     g.x>..i?..~.<...
1c0101d0:	2794 3f0d dba3 3f7f 0dc3 3f0c b5c5 3f61     .'.?...?...?..a?
1c0101e0:	c47b 3f7d 24c5 3f67 15e1 bf65 53b6 be8f     {.}?.$g?..e..S..
1c0101f0:	b6e1 bf0d 3bec 3f79 881d 3f57 181b bea7     .....;y?..W?....
1c010200:	99ab bf5a 3fe5 3f6b 99d2 3dcc d0d7 be20     ..Z..?k?...=.. .
1c010210:	8853 bf6b 7a6c 3f77 8ed3 3ee4 ba0d 3f5e     S.k.lzw?...>..^?
1c010220:	7223 bf5e a1a4 3f57 753a 3dc4 f7b2 3f22     #r^...W?:u.=.."?
1c010230:	0f52 3ef3 4951 bf75 4922 bf77 0f00 3f2b     R..>QIu."Iw...+?
1c010240:	9e98 bd21 f944 bf7e 6f92 be2d c941 3f7e     ..!.D.~..o-.A.~?
1c010250:	ca52 bf61 557b bf26 8f84 bf1b 59e3 3f35     R.a.{U&......Y5?
1c010260:	21b8 bf32 47ea 3e0f d827 3ea6 70b9 3f2c     .!2..G.>'..>.p,?
1c010270:	7c27 3f4a c35d 3ea6 406b bd64 a75a bed1     '|J?]..>k@d.Z...
1c010280:	99ff 3e7d 77a4 bf32 da7c 3f7f 2397 bd29     ..}>.w2.|..?.#).
1c010290:	8380 bf79 70d6 bf1e 4728 3f46 86d8 3f45     ..y..p..(GF?..E?
1c0102a0:	47c1 3e35 49b7 3f77 1171 bf36 5adb bf71     .G5>.Iw?q.6..Zq.
1c0102b0:	c521 bdd8 5c92 3f25 c901 3c6c b5d7 3f0c     !....\%?..l<...?
1c0102c0:	30ee bf71 b910 3f6b 6c74 3f7e ba01 3cab     .0q...k?tl~?...<
1c0102d0:	9fac 3f7d 10ad bf2d 4183 3f5d 0222 bf7f     ..}?..-..A]?"...
1c0102e0:	7a1c 3d16 c988 bf7f b86e 3f17 019d 3f7c     .z.=....n..?..|?
1c0102f0:	a265 3efa 724a beb9 a695 be95 09d3 be8e     e..>Jr..........
1c010300:	ffff bf7f b91b 3f53 15fe bf65 27a1 3e9e     ......S?..e..'.>
1c010310:	b3d5 3f50 8edb 3f57 f40f bf7f b140 3f11     ..P?..W?....@..?
1c010320:	2adc 3e95 6ba0 3e99 a4a7 bf44 87dd bf37     .*.>.k.>..D...7.
1c010330:	cac9 be23 3100 3f4d 200e 3f77 d5ed bf0a     ..#..1M?. w?....
1c010340:	4730 3f7c 0260 3f3e 3dcd bf41 6e07 3f36     0G|?`.>?.=A..n6?
1c010350:	829d 3f16 90cc bf1d ed7c bf50 a2c4 3ef8     ...?....|.P....>
1c010360:	7f5c bf7e dc4f bf7d 5c05 bf7e cdb5 3f3a     \.~.O.}..\~...:?
1c010370:	87c9 bec4 5a33 bf38 6ca2 3eb0 12f7 3f77     ....3Z8..l.>..w?
1c010380:	1e0f 3f39 c3d3 bf14 7faf 3f25 c988 bf22     ..9?......%?..".
1c010390:	f866 3f7b 9b48 be45 74fc 3e64 6710 bed7     f.{?H.E..td>.g..
1c0103a0:	941e beb4 3014 3f05 6e36 3f7d abc9 3ed6     .....0.?6n}?...>
1c0103b0:	65f1 bf6c 0fe5 3efe 5500 3f68 f42f bec1     .el....>.Uh?/...
1c0103c0:	2cb4 bf1b d667 bf1c c0a7 3f62 a057 3dc0     .,..g.....b?W..=
1c0103d0:	f6db 3f56 c9d4 bf7f 1297 bf7e 9e36 3f10     ..V?......~.6..?
1c0103e0:	8aa6 3eeb 5a10 3f60 e349 3f16 847b bf4e     ...>.Z`?I..?{.N.
1c0103f0:	5983 be71 edf6 bedf e7a7 bf4c ff95 becb     .Yq.......L.....
1c010400:	59a1 be32 030c 3f65 dee2 bdb6 35b5 3f41     .Y2...e?.....5A?
1c010410:	8d38 be95 a251 3f1b 8de1 bf52 9cf1 bd39     8...Q..?..R...9.
1c010420:	1660 3f15 215b 3e92 f316 bf4a 4844 3f06     `..?[!.>..J.DH.?
1c010430:	ce69 bf78 bc88 be94 edec bf6c 3d14 bef6     i.x.......l..=..
1c010440:	8eb6 3f62 440d bef8 9f91 bf43 463b bddb     ..b?.D....C.;F..
1c010450:	0c55 be8c fe4e bf62 78b4 3f74 125d 3f6c     U...N.b..xt?].l?
1c010460:	f054 3f78 76a4 3f6f 49bb 3f62 8d8c bf72     T.x?.vo?.Ib?..r.

1c010470 <gm_sin>:
1c010470:	b260 bf52 eedf 3f4c dec7 bf37 0471 beea     `.R...L?..7.q...
1c010480:	1706 3f0a 7ad9 becc 1274 3f66 de3b be5a     ...?.z..t.f?;.Z.
1c010490:	feda 3f7f c10a 3f67 8ed6 bf56 5ca6 be17     ...?..g?..V..\..
1c0104a0:	3aa1 3f6e cf65 3f7f cb78 3f35 eea2 bf7b     .:n?e..?x.5?..{.
1c0104b0:	e191 3f7e a602 3d00 986e 3f6d 279f 3eb9     ..~?...=n.m?.'.>
1c0104c0:	0e04 3f29 7ccd bf05 3831 3d44 1650 bf62     ..)?.|..18D=P.b.
1c0104d0:	f882 bef0 dee9 3f14 1db2 bde1 4fc4 bf7f     .......?.....O..
1c0104e0:	558f bf78 68c9 bed1 2970 3dcc 8c46 3c45     .Ux..h..p).=F.E<
1c0104f0:	9176 bf55 6e4e bd08 4aaf 3f56 9615 bef1     v.U.Nn...JV?....
1c010500:	f0b3 3e06 140e 3edc 8641 3ee4 c3cb 3f75     ...>...>A..>..u?
1c010510:	327a 3f55 de19 be69 2350 bf0a fbb5 3f71     z2U?..i.P#....q?
1c010520:	3a95 3f05 ed50 bec9 b823 3f7e d2c7 3f7c     .:.?P...#.~?..|?
1c010530:	9a7b 3ec8 011b be83 13bd 3f65 6b2b befc     {..>......e?+k..
1c010540:	6855 3efd fb74 bf09 d1c6 3f7e 6d69 3f45     Uh.>t.....~?imE?
1c010550:	506e bf61 918e be92 735f 3e84 75b9 3f3e     nPa....._s.>.u>?
1c010560:	ccf6 bf7f 3383 bdb7 4d29 3f7c 3392 3dc7     .....3..)M|?.3.=
1c010570:	4937 3ef1 9984 3f42 5095 3f4b afdc bf34     7I.>..B?.PK?..4.
1c010580:	dc92 bf37 7b49 3f7d 06bd 3f72 35b3 3f3d     ..7.I{}?..r?.5=?
1c010590:	a3a2 3f1c 0a52 3f72 9a2a bf7f 8dbf bf69     ...?R.r?*.....i.
1c0105a0:	0665 3f78 892d 3f37 9343 bd0a c81a 3f7f     e.x?-.7?C......?
1c0105b0:	0c33 be65 1412 3f49 ee80 3f21 d8de 3f22     3.e...I?..!?.."?
1c0105c0:	f4df bf7b 6f04 be84 f6e5 3f33 b0c5 beaa     ..{..o....3?....
1c0105d0:	8fdd 3f7e 6d33 bf43 f928 3f7f dc77 3f55     ..~?3mC.(..?w.U?
1c0105e0:	9d2f 3eab b4f9 3ec7 ecbe 3de2 f199 bf7f     /..>...>...=....
1c0105f0:	32cc 3e0b a370 bf3c c53b 3f00 16ab bdb4     .2.>p.<.;..?....
1c010600:	d3c2 3f7f f626 bd26 3240 bf4e 2b77 3e34     ...?&.&.@2N.w+4>
1c010610:	3add bf5f 9e21 bf6e d218 bf74 f3a7 bf75     .:_.!.n...t...u.
1c010620:	0771 39ab e84c bf0f 85cc bee4 7b04 bf73     q..9L........{s.
1c010630:	410c 3f14 18c9 bf0a 618d bc9c 7faf bf52     .A.?.....a....R.
1c010640:	e4f8 bf74 3c96 bf74 e9b9 bf23 78fe 3f32     ..t..<t...#..x2?
1c010650:	b432 bf7c 13a5 3f19 a4ac be85 153b 3f57     2.|....?....;.W?
1c010660:	fa5f 3e2d 8f17 3f2b e8f3 bf27 990a 3f33     _.->..+?..'...3?
1c010670:	14d2 bf4f c3d1 bf49 efc2 3f13 c9b9 bf5f     ..O...I....?.._.
1c010680:	8fd4 3ddd 1bf5 be04 7c1e 3de7 0b4a bf2f     ...=.....|.=J./.
1c010690:	63b0 bf6c 9fb2 bf31 5268 bf70 0557 3e86     .cl...1.hRp.W..>
1c0106a0:	d388 3f30 56b2 bf50 4f78 bf43 937b 3f45     ..0?.VP.xOC.{.E?
1c0106b0:	f93b be34 3029 3f7b 8c2a 3f79 3e00 bf68     ;.4.)0{?*.y?.>h.
1c0106c0:	8c9c bf6f a012 bf5a b812 be10 6955 bf68     ..o...Z.....Uih.
1c0106d0:	7cef 3ec4 4255 3f5e 03c3 3ed7 ebbb 3f6c     .|.>UB^?...>..l?
1c0106e0:	9c0a 3f4b 54d7 bf4a a631 3eed dd79 3f7e     ..K?.TJ.1..>y.~?
1c0106f0:	04ef bf0b 8242 3d26 d604 bdfa 3d03 3f53     ....B.&=.....=S?
1c010700:	4d39 3f63 984f bef6 ce6c 3f4e 486d bf17     9Mc?O...l.N?mH..
1c010710:	c9b1 bf78 3791 bf66 75c5 3f19 cd6a bf6a     ..x..7f..u.?j.j.
1c010720:	1652 bf7c d1b3 3ee4 fa37 3f7e f244 bf27     R.|....>7.~?D.'.
1c010730:	d5f8 3f74 4231 3f4b 9cbb bf11 bcad 3f7f     ..t?1BK?.......?
1c010740:	1ba7 bf50 5a0d bf75 096b bf1c f46a 3f59     ..P..Zu.k...j.Y?
1c010750:	0b9e 3e71 f5be bf74 e978 3ec1 731b bf60     ..q>..t.x..>.s`.
1c010760:	646b beee e401 bf5f 20f9 3f25 8748 bf7e     kd...._.. %?H.~.
1c010770:	3ca8 bf76 ba4d beec eabd be97 0d47 bec6     .<v.M.......G...
1c010780:	d8bc be6e 0883 beb5 69ec beef c1c8 bea3     ..n......i......

1c010790 <pos_uart>:
	...
